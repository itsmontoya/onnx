package onnx

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/operators"
)

// BenchmarkMNISTNodes isolates prepared calls using real intermediate tensors.
// It excludes session dispatch and destination allocation, and does not retire
// values so that each node retains its original, disjoint inputs and output.
func BenchmarkMNISTNodes(b *testing.B) {
	var (
		ctx     = context.Background()
		model   *Model
		session *Session
		input   *Tensor
		run     *evaluation
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

	if run, err = newEvaluation(ctx, session.plan); err != nil {
		b.Fatal(err)
	}

	if err = run.bind(map[string]*Tensor{"Input3": input}); err != nil {
		b.Fatal(err)
	}

	for _, step := range session.plan.steps {
		if err = run.execute(step); err != nil {
			b.Fatal(err)
		}

		if operators.IsShapeOperator(step.Operator) || step.Operator == "Constant" || step.Operator == "Identity" {
			continue
		}

		inputs := make([]*tensor.Tensor, len(step.Inputs))
		for slot, valueIndex := range step.Inputs {
			if valueIndex >= 0 {
				inputs[slot] = run.storage.values[valueIndex]
			}
		}
		output := run.storage.values[step.Outputs[0]]
		b.Run(fmt.Sprintf("%02d_%s", step.Index, step.Operator), func(node *testing.B) {
			var (
				cnn       *cnnCall
				numerical *numericalCall
				nodeErr   error
			)

			if operators.IsCNNOperator(step.Operator) {
				if cnn, nodeErr = newCNNCall(step, inputs, session.plan.limits); nodeErr != nil {
					node.Fatal(nodeErr)
				}

				node.Logf("input=%v output=%v conv=%+v pool=%+v", inputs[0].Shape(), output.Shape(), cnn.conv, cnn.pool)
				node.ReportAllocs()
				node.ResetTimer()
				for node.Loop() {
					if _, nodeErr = cnn.run(ctx, output); nodeErr != nil {
						node.Fatal(nodeErr)
					}
				}
				return
			}

			if numerical, nodeErr = newNumericalCall(step, inputs, session.plan.limits); nodeErr != nil {
				node.Fatal(nodeErr)
			}

			node.Logf("left=%v output=%v", inputs[0].Shape(), output.Shape())
			if step.Operator != "Relu" {
				node.ReportAllocs()
				node.ResetTimer()
				for node.Loop() {
					if _, nodeErr = numerical.run(ctx, output); nodeErr != nil {
						node.Fatal(nodeErr)
					}
				}
				return
			}

			node.Run("WithCorrection", func(complete *testing.B) {
				complete.ReportAllocs()
				for complete.Loop() {
					if _, nodeErr = numerical.run(ctx, output); nodeErr != nil {
						complete.Fatal(nodeErr)
					}
				}
			})
			node.Run("BackendOnly", func(kernel *testing.B) {
				kernel.ReportAllocs()
				for kernel.Loop() {
					if nodeErr = tensor.ReLUInto(inputs[0], output); nodeErr != nil {
						kernel.Fatal(nodeErr)
					}
				}
			})
			node.Run("ONNXCorrectionOnly", func(correction *testing.B) {
				correction.ReportAllocs()
				for correction.Loop() {
					if _, nodeErr = numerical.patchRelu(ctx, output); nodeErr != nil {
						correction.Fatal(nodeErr)
					}
				}
			})
		})
		if step.Operator == "Conv" && step.Index == 4 {
			b.Run("04_ConvRotatingWorkingSet", func(cache *testing.B) {
				var call *cnnCall
				if call, err = newCNNCall(step, inputs, session.plan.limits); err != nil {
					cache.Fatal(err)
				}

				benchmarkMNISTConvCache(cache, inputs[0], inputs[1], output.Shape(), call.conv)
			})
		}
	}
}

// benchmarkMNISTConvCache rotates independent, identical tensor sets to probe
// cache sensitivity. These are separate diagnostics, not the MNIST workload.
func benchmarkMNISTConvCache(b *testing.B, input, weights *tensor.Tensor, outputShape []int64, config tensor.Conv2DConfig) {
	var (
		inputValues  []float32
		weightValues []float32
		err          error
	)

	if inputValues, err = input.Float32Values(); err != nil {
		b.Fatal(err)
	}

	if weightValues, err = weights.Float32Values(); err != nil {
		b.Fatal(err)
	}

	inputShape, weightShape := input.Shape(), weights.Shape()
	for _, count := range []int{1, 64, 1024} {
		b.Run(fmt.Sprintf("%04dSets", count), func(rotation *testing.B) {
			var (
				inputSets  = make([]*tensor.Tensor, count)
				weightSets = make([]*tensor.Tensor, count)
				outputs    = make([]*tensor.Tensor, count)
				position   int
				kernelErr  error
			)

			for index := range inputSets {
				if inputSets[index], kernelErr = tensor.FromFloat32(inputShape, inputValues); kernelErr != nil {
					rotation.Fatal(kernelErr)
				}

				if weightSets[index], kernelErr = tensor.FromFloat32(weightShape, weightValues); kernelErr != nil {
					rotation.Fatal(kernelErr)
				}

				if outputs[index], kernelErr = tensor.New(Float32, outputShape); kernelErr != nil {
					rotation.Fatal(kernelErr)
				}
			}
			rotation.Logf("independent payload bytes=%d", int64(count)*(inputSets[0].ByteSize()+weightSets[0].ByteSize()+outputs[0].ByteSize()))
			rotation.ReportAllocs()
			rotation.ResetTimer()
			for rotation.Loop() {
				if kernelErr = tensor.Conv2DInto(inputSets[position], weightSets[position], nil, outputs[position], config); kernelErr != nil {
					rotation.Fatal(kernelErr)
				}

				position++
				if position == count {
					position = 0
				}
			}
		})
	}
}
