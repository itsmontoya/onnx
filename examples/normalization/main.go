// Command normalization runs an unchanged Apache-2.0 upstream normalization export.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/itsmontoya/onnx"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		model          *onnx.Model
		session        *onnx.Session
		x, scale, bias *onnx.Tensor
		outputs        []onnx.NamedTensor
	)

	ctx := context.Background()
	if model, err = onnx.LoadFile(ctx, "testdata/corpus/models/test_layer_normalization_default_axis.onnx", onnx.Options{}); err != nil {
		return fmt.Errorf("load normalization model: %w", err)
	}

	defer func() { err = errors.Join(err, model.Close()) }()
	values := make([]float32, 120)
	for index := range values {
		values[index] = float32(index) / 8
	}
	if x, err = onnx.NewFloat32([]int64{2, 3, 4, 5}, values); err != nil {
		return fmt.Errorf("construct features: %w", err)
	}

	if scale, err = onnx.NewFloat32([]int64{5}, []float32{1, 1, 1, 1, 1}); err != nil {
		return fmt.Errorf("construct scale: %w", err)
	}

	if bias, err = onnx.NewFloat32([]int64{5}, []float32{0, 0, 0, 0, 0}); err != nil {
		return fmt.Errorf("construct bias: %w", err)
	}

	if session, err = model.NewSession(); err != nil {
		return fmt.Errorf("create normalization session: %w", err)
	}

	defer func() { err = errors.Join(err, session.Close()) }()
	if outputs, err = session.Run(ctx, map[string]*onnx.Tensor{"X": x, "W": scale, "B": bias}, onnx.RunOptions{}); err != nil {
		return fmt.Errorf("run normalization: %w", err)
	}

	metadata := model.Metadata()
	fmt.Printf("%s: IR %d, opset %d\n", metadata.Name, metadata.IRVersion, metadata.Opset)
	for _, output := range outputs {
		var data []float32
		if data, err = output.Tensor.Float32Values(); err != nil {
			return fmt.Errorf("read output %q: %w", output.Name, err)
		}

		fmt.Printf("%s %v first = %.6f\n", output.Name, output.Tensor.Shape(), data[0])
	}
	return nil
}
