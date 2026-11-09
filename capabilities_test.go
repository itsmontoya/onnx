package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"testing"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/operators"
)

func TestCapabilitiesCanonicalManifest(t *testing.T) {
	records := Capabilities()
	if len(records) != 25 {
		t.Fatalf("capability count = %d", len(records))
	}

	implemented := 0
	for _, record := range records {
		if record.MinOpset != 13 || record.MaxOpset != 13 {
			t.Fatalf("untested interval: %+v", record)
		}

		if record.Status == "implemented" {
			implemented++
			if record.Operator == "Identity" && record.Task != "RT-07" || record.Operator != "Identity" && record.Task != "RT-09" && record.Task != "RT-12" && record.Task != "RT-13" {
				t.Fatalf("unexpected implemented operator: %+v", record)
			}
		}

		if record.Operator == "Constant" && (len(record.Attributes) != 5 || record.Attributes["value"].Required) {
			t.Fatalf("Constant forms differ: %+v", record)
		}
	}

	if implemented != 25 {
		t.Fatalf("implemented count = %d", implemented)
	}

	data, err := os.ReadFile("capabilities.json")
	if err != nil {
		t.Fatal(err)
	}

	if string(data) != string(capabilityJSON) || !json.Valid(data) {
		t.Fatal("embedded capabilities differ from canonical file")
	}
}

func TestCapabilitiesDeepCopies(t *testing.T) {
	first := Capabilities()
	for index := range first {
		first[index].Operator = "mutated"
		first[index].DataTypes[0] = "mutated"
		first[index].Restrictions[0] = "mutated"
		if len(first[index].Inputs.OptionalSlots) > 0 {
			first[index].Inputs.OptionalSlots[0] = 99
		}

		if len(first[index].Outputs.OptionalSlots) > 0 {
			first[index].Outputs.OptionalSlots[0] = 99
		}

		for name, attribute := range first[index].Attributes {
			if len(attribute.Default) > 0 {
				attribute.Default[0] = '9'
			}

			delete(first[index].Attributes, name)
		}
	}

	second := Capabilities()
	if second[0].Operator != "Identity" || second[0].DataTypes[0] != "float32" || second[1].Attributes["value"].Type != "TENSOR" {
		t.Fatal("returned records mutated canonical manifest")
	}

	for _, record := range second {
		if record.Operator == "Softmax" && string(record.Attributes["axis"].Default) != "-1" {
			t.Fatal("returned default shared storage")
		}
	}

	var wait sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for repeat := 0; repeat < 5; repeat++ {
				Capabilities()[0].DataTypes[0] = "local"
			}
		}()
	}
	wait.Wait()
}

func TestCompileEmbeddedProfile(t *testing.T) {
	data, err := os.ReadFile("testdata/profile/identity.onnx")
	if err != nil {
		t.Fatal(err)
	}

	wire, err := decode.ParseModel(context.Background(), data, decode.Limits{})
	if err != nil {
		t.Fatal(err)
	}

	source, err := graph.Build(context.Background(), wire, int64(len(data)), graph.Limits{})
	if err != nil {
		t.Fatal(err)
	}

	plan, err := compileGraph(context.Background(), source, operators.Limits{})
	if err != nil || len(plan.Steps()) != 1 || plan.Steps()[0].Operator != "Identity" {
		t.Fatalf("profile compilation = %v, %v", plan, err)
	}

	_, err = compileGraph(context.Background(), source, operators.Limits{MaxLoadBytes: 1})
	var diagnostic *Error
	if !errors.Is(err, ErrLimit) || !errors.As(err, &diagnostic) {
		t.Fatalf("public compilation errors = %v", err)
	}

	if _, err = compileGraph(nil, source, operators.Limits{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("public argument sentinel = %v", err)
	}
}
