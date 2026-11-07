// Command inference runs the checked-in dynamic-batch MLP or branching graph.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"path/filepath"

	"github.com/itsmontoya/onnx"
)

func main() {
	graph := flag.String("graph", "mlp", "fixture graph: mlp or branch")
	batch := flag.Int("batch", 1, "fixture batch size: 1 or 3")
	flag.Parse()
	if err := run(*graph, *batch); err != nil {
		log.Fatal(err)
	}
}

func run(graph string, batch int) (err error) {
	var (
		model   *onnx.Model
		session *onnx.Session
		inputs  map[string]*onnx.Tensor
		outputs []onnx.NamedTensor
	)

	if inputs, err = fixtureInputs(graph, batch); err != nil {
		return err
	}

	ctx := context.Background()
	path := filepath.Join("testdata", "inference", "models", graph+".onnx")
	if model, err = onnx.LoadFile(ctx, path, onnx.Options{}); err != nil {
		return fmt.Errorf("load %s: %w", graph, err)
	}

	defer func() { err = errors.Join(err, model.Close()) }()
	metadata := model.Metadata()
	fmt.Printf("%s: IR %d, opset %d, %d input(s), %d output(s)\n", metadata.Name, metadata.IRVersion, metadata.Opset, len(metadata.Inputs), len(metadata.Outputs))
	for _, input := range metadata.Inputs {
		fmt.Printf("input %s: rank %d, leading symbol %q\n", input.Name, len(input.Dimensions), input.Dimensions[0].Symbol)
	}
	if session, err = model.NewSession(); err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	defer func() { err = errors.Join(err, session.Close()) }()
	if outputs, err = session.Run(ctx, inputs, onnx.RunOptions{}); err != nil {
		return fmt.Errorf("run %s batch %d: %w", graph, batch, err)
	}

	for _, output := range outputs {
		var values []float32
		if values, err = output.Tensor.Float32Values(); err != nil {
			return fmt.Errorf("read output %q: %w", output.Name, err)
		}

		fmt.Printf("%s %v = %.6f\n", output.Name, output.Tensor.Shape(), values)
	}
	return nil
}

func fixtureInputs(graph string, batch int) (inputs map[string]*onnx.Tensor, err error) {
	if graph != "mlp" && graph != "branch" {
		return nil, fmt.Errorf("graph must be mlp or branch")
	}

	if batch != 1 && batch != 3 {
		return nil, fmt.Errorf("fixture batch must be 1 or 3")
	}

	var (
		features = []float32{1, 2, -1, 0.5, 0, -2}
		residual = []float32{0.25, -0.5, 1, 0, -0.25, 0.75}
		shape    = []int64{int64(batch), 2}
		value    *onnx.Tensor
	)

	if value, err = onnx.NewFloat32(shape, features[:batch*2]); err != nil {
		return nil, fmt.Errorf("create features: %w", err)
	}

	inputs = map[string]*onnx.Tensor{"features": value}
	if graph == "branch" {
		if value, err = onnx.NewFloat32(shape, residual[:batch*2]); err != nil {
			return nil, fmt.Errorf("create residual: %w", err)
		}

		inputs["residual"] = value
	}

	return inputs, nil
}
