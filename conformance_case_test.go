package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"github.com/itsmontoya/onnx/internal/operators"
	"github.com/itsmontoya/onnx/internal/profile"
)

type conformanceCase struct {
	Name      string           `json:"name"`
	Origin    string           `json:"origin"`
	License   string           `json:"license"`
	Status    string           `json:"status"`
	Reason    string           `json:"reason"`
	IRVersion int64            `json:"ir_version"`
	Opsets    map[string]int64 `json:"opsets"`
	Schemas   []struct {
		Domain   string `json:"domain"`
		Operator string `json:"operator"`
		Version  int64  `json:"version"`
	} `json:"schemas"`
	Model        conformanceArtifact        `json:"model"`
	Comparison   conformanceComparison      `json:"comparison"`
	Inputs       []conformanceValue         `json:"inputs"`
	Outputs      []conformanceValue         `json:"outputs"`
	Observations map[string]json.RawMessage `json:"observations"`
}

func (c conformanceCase) run() (outputs []NamedTensor, err error) {
	var (
		data    []byte
		inputs  map[string]*Tensor
		model   *Model
		session *Session
	)

	if data, err = c.Model.read(); err != nil {
		return nil, err
	}

	if err = c.validateEnvelope(data); err != nil {
		return nil, err
	}

	if inputs, err = c.readInputs(); err != nil {
		return nil, err
	}

	model, err = LoadBytes(context.Background(), data, Options{})
	if c.Status == "unsupported" {
		if model != nil || !errors.Is(err, ErrUnsupported) {
			return nil, fmt.Errorf("case %q inputs %v: expected unsupported (%s), got model=%v error=%v", c.Name, c.inputNames(), c.Reason, model, err)
		}

		// Check expected values' hashes/decoding even though no numerical support is claimed.
		for _, value := range c.Outputs {
			if _, err = value.read(); err != nil {
				return nil, fmt.Errorf("unsupported case output: %w", err)
			}
		}
		return nil, nil
	}

	if err != nil {
		return nil, fmt.Errorf("load case %q inputs %v: %w", c.Name, c.inputNames(), err)
	}

	defer func() { err = errors.Join(err, model.Close()) }()
	if session, err = model.NewSession(); err != nil {
		return nil, fmt.Errorf("session for case %q: %w", c.Name, err)
	}

	defer func() { err = errors.Join(err, session.Close()) }()
	if outputs, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
		return nil, fmt.Errorf("run case %q inputs %v: %w", c.Name, c.inputNames(), err)
	}

	if err = c.compareOutputs(outputs); err != nil {
		return nil, err
	}

	return outputs, nil
}

func (c conformanceCase) validateEnvelope(data []byte) (err error) {
	var wire *onnxpb.ModelProto
	if wire, err = decode.ParseModel(context.Background(), data, decode.Limits{}); err != nil {
		return fmt.Errorf("case %q model: %w", c.Name, err)
	}

	imports := make(map[string]int64)
	for _, value := range wire.OpsetImport {
		imports[value.GetDomain()] = value.GetVersion()
	}
	if wire.GetIrVersion() != c.IRVersion || !maps.Equal(imports, c.Opsets) {
		return fmt.Errorf("case %q IR/import provenance differs", c.Name)
	}

	if len(wire.GetGraph().GetNode()) != len(c.Schemas) {
		return fmt.Errorf("case %q schema count differs", c.Name)
	}

	var registry *operators.Registry
	if registry, err = operators.NewRegistry(Capabilities()); err != nil {
		return err
	}

	for index, node := range wire.Graph.Node {
		schema := c.Schemas[index]
		if node.GetDomain() != schema.Domain || node.GetOpType() != schema.Operator || schema.Version <= 0 || schema.Version > imports[schema.Domain] {
			return fmt.Errorf("case %q node %d schema provenance differs", c.Name, index)
		}

		if profile.Accepts(c.IRVersion, schema.Domain, imports[schema.Domain]) && c.Status != "unsupported" {
			var selected Capability
			if selected, err = registry.Resolve(schema.Domain, schema.Operator, imports[schema.Domain]); err != nil {
				return fmt.Errorf("case %q schema selection: %w", c.Name, err)
			}

			if selected.SchemaVersion != schema.Version {
				return fmt.Errorf("case %q node %d applicable schema differs", c.Name, index)
			}
		}
	}
	return nil
}

func (c conformanceCase) readInputs() (inputs map[string]*Tensor, err error) {
	inputs = make(map[string]*Tensor)
	for _, value := range c.Inputs {
		var decoded NamedTensor
		if decoded, err = value.read(); err != nil {
			return nil, fmt.Errorf("case %q input %q: %w", c.Name, value.Name, err)
		}

		if _, exists := inputs[decoded.Name]; exists {
			return nil, fmt.Errorf("case %q duplicate input %q", c.Name, decoded.Name)
		}

		inputs[decoded.Name] = decoded.Tensor
	}
	return inputs, nil
}

func (c conformanceCase) compareOutputs(outputs []NamedTensor) (err error) {
	if len(outputs) != len(c.Outputs) {
		return fmt.Errorf("case %q inputs %v: output count %d, want %d", c.Name, c.inputNames(), len(outputs), len(c.Outputs))
	}

	for index, value := range c.Outputs {
		var expected NamedTensor
		if expected, err = value.read(); err != nil {
			return fmt.Errorf("case %q output %q: %w", c.Name, value.Name, err)
		}

		if err = c.Comparison.compare("output", outputs[index], expected); err != nil {
			return fmt.Errorf("case %q inputs %v: %w", c.Name, c.inputNames(), err)
		}
	}
	return nil
}

func (c conformanceCase) inputNames() (names []string) {
	for _, value := range c.Inputs {
		names = append(names, value.Name)
	}
	return names
}
