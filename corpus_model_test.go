package onnx

import (
	"context"
	"errors"
	"maps"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

type corpusModel struct {
	Name         string              `json:"name"`
	Origin       string              `json:"origin"`
	SourcePath   string              `json:"source_path"`
	SourceSHA256 string              `json:"source_sha256"`
	License      string              `json:"license"`
	Model        conformanceArtifact `json:"model"`
	IRVersion    int64               `json:"ir_version"`
	Opsets       map[string]int64    `json:"opsets"`
	Status       string              `json:"status"`
	Before       struct {
		Count        int   `json:"count"`
		FailingNodes []int `json:"failing_nodes"`
	} `json:"before_rt19"`
	After struct {
		Count        int   `json:"count"`
		FailingNodes []int `json:"failing_nodes"`
	} `json:"after_rt19"`
	Nodes []struct {
		Index      int      `json:"index"`
		Name       string   `json:"name"`
		Domain     string   `json:"domain"`
		Operator   string   `json:"operator"`
		Inputs     []string `json:"inputs"`
		Outputs    []string `json:"outputs"`
		Attributes map[string]struct {
			Type string `json:"type"`
		} `json:"attributes"`
	} `json:"nodes"`
}

func (m corpusModel) check(t *testing.T) {
	t.Helper()
	data, err := m.Model.read()
	if err != nil {
		t.Fatal(err)
	}

	if m.SourceSHA256 != m.Model.SHA256 || m.License != "Apache-2.0" || !strings.HasPrefix(m.Origin, "https://github.com/onnx/onnx/blob/3bf92c03a9f27eba3bda1e5b9e63ea20ec213557/") || !strings.HasSuffix(m.Origin, m.SourcePath) {
		t.Fatal("corpus source/license provenance differs")
	}

	var wire onnxpb.ModelProto
	if err = proto.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}

	imports := make(map[string]int64)
	for _, imported := range wire.OpsetImport {
		imports[imported.GetDomain()] = imported.GetVersion()
	}
	if m.IRVersion != wire.GetIrVersion() || !maps.Equal(m.Opsets, imports) || len(m.Nodes) != len(wire.Graph.Node) {
		t.Fatal("corpus metadata differs from original model bytes")
	}

	for index, node := range wire.Graph.Node {
		recorded := m.Nodes[index]
		if recorded.Index != index || recorded.Name != node.GetName() || recorded.Domain != node.GetDomain() || recorded.Operator != node.GetOpType() || len(recorded.Attributes) != len(node.Attribute) {
			t.Fatalf("node %d corpus metadata differs", index)
		}

		for _, attribute := range node.Attribute {
			if recorded.Attributes[attribute.GetName()].Type != attribute.GetType().String() {
				t.Fatalf("node %d attribute %q metadata differs", index, attribute.GetName())
			}
		}
	}
	model, loadErr := LoadBytes(context.Background(), data, Options{})
	if m.Status == "enabled" {
		if loadErr != nil || model == nil || m.Before.Count != 1 || len(m.Before.FailingNodes) != 1 || m.Before.FailingNodes[0] != 0 || m.After.Count != 0 {
			t.Fatalf("enabled corpus model = %v, %v; reports = %+v / %+v", model, loadErr, m.Before, m.After)
		}

		if err = model.Close(); err != nil {
			t.Fatal(err)
		}

		return
	}

	if model != nil || !errors.Is(loadErr, ErrUnsupported) || m.Status != "unsupported" || m.After.Count == 0 {
		t.Fatalf("corpus negative = %v, %v; report = %+v", model, loadErr, m.After)
	}
}
