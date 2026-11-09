package onnx

import (
	"context"
	"errors"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestRunStorageZeroContractionOverwritesReusedDestination(t *testing.T) {
	wire := modelWire(t, "m1/matmul_empty_contraction")
	input := proto.Clone(wire.Graph.Output[0]).(*onnxpb.ValueInfoProto)
	input.Name = proto.String("x")
	wire.Graph.Input = append(wire.Graph.Input, input)
	wire.Graph.Node = append([]*onnxpb.NodeProto{{OpType: proto.String("Add"), Input: []string{"x", "x"}, Output: []string{"unused"}}}, wire.Graph.Node...)
	plan := loadedModel(t, modelBytes(t, wire), Options{}).plan
	run, err := newEvaluation(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}

	outputs, runErr := run.run(numericalInputs(t, wire), RunOptions{})
	if runErr != nil || run.storage.reuses != 1 {
		t.Fatalf("zero-contraction reuse: %d, %v", run.storage.reuses, runErr)
	}

	assertFloats(t, outputs[0].Tensor, []int64{2, 3}, []float32{0, 0, 0, 0, 0, 0})
}

func TestRunStorageReusesOnlyAfterEveryAliasConsumer(t *testing.T) {
	plan := loadedModel(t, modelBytes(t, reuseWire(t)), Options{}).plan
	run, err := newEvaluation(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}

	input := floatTensor(t, []int64{1, 2}, []float32{1, 2})
	if err = run.bind(map[string]*Tensor{"x": input}); err != nil {
		t.Fatal(err)
	}

	left, alias, right, join := plan.indices["a"], plan.indices["alias"], plan.indices["b"], plan.indices["join"]
	for position := 0; position <= 3; position++ {
		if err = run.execute(plan.steps[position]); err != nil {
			t.Fatal(err)
		}

		if position == 1 && run.storage.values[alias] != run.storage.values[left] {
			t.Fatal("Identity did not retain the same backing value")
		}

		if position == 2 && (run.storage.values[right] == input.value || run.storage.values[right] == run.storage.values[left]) {
			t.Fatal("a still-live input or alias was reused")
		}

		run.storage.after(position)
	}
	if run.storage.values[left] != nil || run.storage.values[alias] != nil || run.storage.free == nil {
		t.Fatal("last use did not retire every alias slot")
	}

	if run.budget.live != 24 {
		t.Fatalf("replaced free candidate was not released: %d", run.budget.live)
	}

	first := run.storage.free
	if err = run.execute(plan.steps[4]); err != nil {
		t.Fatal(err)
	}

	current := run.storage.values[plan.indices["c"]]
	if current != first || current == run.storage.values[join] {
		t.Fatal("destination reuse did not preserve disjoint live arguments")
	}

	run.storage.after(4)
	if err = run.execute(plan.steps[5]); err != nil {
		t.Fatal(err)
	}

	run.storage.after(5)
	run.storage.trim()
	outputs, outputErr := run.publish(plan.outputIndices)
	if outputErr != nil || run.storage.reuses != 2 || run.storage.free != nil {
		t.Fatalf("reuse/publication: %d, %v", run.storage.reuses, outputErr)
	}

	assertFloats(t, outputs[0].Tensor, []int64{1, 2}, []float32{6, 12})
	assertFloats(t, input, []int64{1, 2}, []float32{1, 2})
}

func TestRunStorageChecksTypeAndEmptyShapeBeforeReuse(t *testing.T) {
	tests := []struct {
		name     string
		source   *Tensor
		dataType DataType
		shape    []int64
		bytes    int64
		reuse    bool
	}{
		{"same bytes different type", floatTensor(t, []int64{2}, []float32{1, 2}), Int64, []int64{1}, 8, false},
		{"empty matching shape", floatTensor(t, []int64{0, 2}, nil), Float32, []int64{0, 2}, 0, true},
		{"empty different shape", floatTensor(t, []int64{0, 2}, nil), Float32, []int64{2, 0}, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			budget, err := newRunBudget(Limits{MaxTensorBytes: 16, MaxLiveBytes: 32, MaxWorkspaceBytes: 16}, tt.source.ByteSize())
			if err != nil {
				t.Fatal(err)
			}

			storage := newRunStorage(&valueLifetimes{owners: []int{0}}, budget)
			storage.free = tt.source.value
			destination, destinationErr := storage.destination(tt.dataType, tt.shape, tt.bytes, 0)
			if destinationErr != nil || (destination == tt.source.value) != tt.reuse || budget.live != tt.bytes {
				t.Fatalf("typed/empty reuse: %v, %+v, %v", destination, budget, destinationErr)
			}
		})
	}
}

func TestRunStorageTrimsFreePayloadBeforeReservations(t *testing.T) {
	tests := []struct {
		name      string
		matching  bool
		planning  bool
		workspace int64
		want      error
	}{
		{"different shape before allocation", false, false, 8, nil},
		{"matching destination", true, false, 8, nil},
		{"planning workspace trims candidate", false, true, 16, nil},
		{"workspace failure still rejects", true, false, 17, ErrLimit},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			budget, err := newRunBudget(Limits{MaxTensorBytes: 16, MaxLiveBytes: 32, MaxWorkspaceBytes: 16}, 8)
			if err != nil {
				t.Fatal(err)
			}

			storage := newRunStorage(&valueLifetimes{owners: []int{0}}, budget)
			candidate := floatTensor(t, []int64{4}, []float32{1, 2, 3, 4})
			if tt.matching {
				candidate = floatTensor(t, []int64{2}, []float32{5, 6})
			}

			storage.free = candidate.value
			if err = budget.reserve(candidate.ByteSize(), 0); err != nil {
				t.Fatal(err)
			}

			workspace := tt.workspace
			if tt.planning {
				if err = storage.reserve(0, workspace); err != nil || storage.free != nil {
					t.Fatalf("planning workspace did not trim free payload: %v", err)
				}

				workspace = 0
			}

			destination, destinationErr := storage.destination(Float32, []int64{2}, 8, workspace)
			if !errors.Is(destinationErr, tt.want) {
				t.Fatalf("reservation error: %v", destinationErr)
			}

			if tt.want != nil {
				if destination != nil || budget.workspace != 0 || budget.live != 8 {
					t.Fatalf("failed reservation accounting: %+v", budget)
				}

				return
			}

			if storage.free != nil || budget.live != 16 || budget.workspace != tt.workspace {
				t.Fatalf("trimmed reservation accounting: %+v", budget)
			}

			if tt.matching && destination != candidate.value || !tt.matching && destination != nil {
				t.Fatal("destination did not match the checked free buffer")
			}
		})
	}
}
