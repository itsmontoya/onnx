// Command external runs the externalized MLP through a portable explicit resolver.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/itsmontoya/onnx"
)

func main() {
	directory := flag.String("directory", "testdata/external/models", "fixture directory")
	batch := flag.Int("batch", 1, "fixture batch size: 1 or 3")
	flag.Parse()
	if err := run(*directory, *batch); err != nil {
		log.Fatal(err)
	}
}

func run(directory string, batch int) (err error) {
	if batch != 1 && batch != 3 {
		return fmt.Errorf("fixture batch must be 1 or 3")
	}

	var (
		weights []byte
		model   *onnx.Model
		session *onnx.Session
		input   *onnx.Tensor
		outputs []onnx.NamedTensor
	)

	if weights, err = os.ReadFile(filepath.Join(directory, "weights.bin")); err != nil {
		return fmt.Errorf("read fixture weights: %w", err)
	}

	ctx := context.Background()
	options := onnx.Options{ExternalData: newMemoryResolver(weights)}
	if model, err = onnx.LoadFile(ctx, filepath.Join(directory, "mlp.onnx"), options); err != nil {
		return fmt.Errorf("load external MLP: %w", err)
	}

	defer func() { err = errors.Join(err, model.Close()) }()
	if session, err = model.NewSession(); err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	defer func() { err = errors.Join(err, session.Close()) }()
	features := []float32{1, 2, -1, 0.5, 0, -2}
	if input, err = onnx.NewFloat32([]int64{int64(batch), 2}, features[:batch*2]); err != nil {
		return fmt.Errorf("create features: %w", err)
	}

	if outputs, err = session.Run(ctx, map[string]*onnx.Tensor{"features": input}, onnx.RunOptions{}); err != nil {
		return fmt.Errorf("run external MLP: %w", err)
	}

	for _, output := range outputs {
		var values []float32
		if values, err = output.Tensor.Float32Values(); err != nil {
			return fmt.Errorf("read %q: %w", output.Name, err)
		}

		fmt.Printf("%s %v = %.6f\n", output.Name, output.Tensor.Shape(), values)
	}
	return nil
}
