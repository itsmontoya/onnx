package onnx

import (
	"context"
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/operators"
)

// BenchmarkMNISTPipeline separates call setup from a prepared, disjoint graph.
// PreparedGraph excludes planning, destination allocation, retirement and output
// publication. These diagnostics do not replace the full BenchmarkMNIST workload.
func BenchmarkMNISTPipeline(b *testing.B) {
	var (
		ctx     = context.Background()
		model   *Model
		session *Session
		input   *Tensor
		err     error
	)

	if model, err = LoadFile(ctx, ".cache/compare-runtime/mnist-ir8-opset13.onnx", Options{}); errors.Is(err, os.ErrNotExist) {
		b.Skip("prepare the MNIST model with tools/compare_runtime.py")
	} else if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() {
		if closeErr := model.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	})
	if session, err = model.NewSession(); err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() {
		if closeErr := session.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	})
	values := make([]float32, 28*28)
	for index := range values {
		values[index] = float32(index%256) / 255
	}
	if input, err = NewFloat32([]int64{1, 1, 28, 28}, values); err != nil {
		b.Fatal(err)
	}

	inputs := map[string]*Tensor{"Input3": input}
	b.Run("Bind", func(binding *testing.B) {
		var (
			run     *evaluation
			bindErr error
		)

		binding.ReportAllocs()
		for binding.Loop() {
			if run, bindErr = newEvaluation(ctx, session.plan); bindErr != nil {
				binding.Fatal(bindErr)
			}

			if bindErr = run.bind(inputs); bindErr != nil {
				binding.Fatal(bindErr)
			}
		}
	})
	b.Run("PreparedGraph", func(prepared *testing.B) {
		var (
			run         *evaluation
			calls       []func() error
			expected    [][]float32
			preparedErr error
		)

		if run, preparedErr = newEvaluation(ctx, session.plan); preparedErr != nil {
			prepared.Fatal(preparedErr)
		}

		if preparedErr = run.bind(inputs); preparedErr != nil {
			prepared.Fatal(preparedErr)
		}

		for _, step := range session.plan.steps {
			if preparedErr = run.execute(step); preparedErr != nil {
				prepared.Fatal(preparedErr)
			}

			if step.Operator == "Identity" || step.Operator == "Constant" {
				continue
			}

			arguments := make([]*tensor.Tensor, len(step.Inputs))
			for slot, valueIndex := range step.Inputs {
				if valueIndex >= 0 {
					arguments[slot] = run.storage.values[valueIndex]
				}
			}
			destination := run.storage.values[step.Outputs[0]]
			if operators.IsCNNOperator(step.Operator) {
				var cnn *cnnCall
				if cnn, preparedErr = newCNNCall(step, arguments, session.plan.limits); preparedErr != nil {
					prepared.Fatal(preparedErr)
				}

				calls = append(calls, func() (callErr error) {
					_, callErr = cnn.run(ctx, destination)
					return callErr
				})
				continue
			}

			if operators.IsShapeOperator(step.Operator) {
				var shape *shapeCall
				if shape, preparedErr = newShapeCall(step, arguments, session.plan.limits); preparedErr != nil {
					prepared.Fatal(preparedErr)
				}

				if preparedErr = shape.plan(ctx); preparedErr != nil {
					prepared.Fatal(preparedErr)
				}

				calls = append(calls, func() (callErr error) {
					_, callErr = shape.run(ctx, destination)
					return callErr
				})
				continue
			}

			var numerical *numericalCall
			if numerical, preparedErr = newNumericalCall(step, arguments, session.plan.limits); preparedErr != nil {
				prepared.Fatal(preparedErr)
			}

			calls = append(calls, func() (callErr error) {
				_, callErr = numerical.run(ctx, destination)
				return callErr
			})
		}
		for _, outputIndex := range session.plan.outputIndices {
			var outputValues []float32
			if outputValues, preparedErr = run.storage.values[outputIndex].Float32Values(); preparedErr != nil {
				prepared.Fatal(preparedErr)
			}

			expected = append(expected, outputValues)
		}
		prepared.ReportAllocs()
		prepared.ResetTimer()
		for prepared.Loop() {
			for _, call := range calls {
				if preparedErr = call(); preparedErr != nil {
					prepared.Fatal(preparedErr)
				}
			}
		}
		prepared.StopTimer()
		for index, outputIndex := range session.plan.outputIndices {
			var actual []float32
			if actual, preparedErr = run.storage.values[outputIndex].Float32Values(); preparedErr != nil {
				prepared.Fatal(preparedErr)
			}

			if !slices.Equal(actual, expected[index]) {
				prepared.Fatal("prepared graph differs from ordinary execution")
			}
		}
	})
}
