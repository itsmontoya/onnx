package onnx

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"strings"
	"testing"
)

func readConformanceSuite() (out *conformanceSuite, err error) {
	var (
		suite conformanceSuite
		data  []byte
	)

	if data, err = os.ReadFile("testdata/conformance/provenance.json"); err != nil {
		return nil, fmt.Errorf("read conformance provenance: %w", err)
	}

	if err = json.Unmarshal(data, &suite); err != nil {
		return nil, fmt.Errorf("parse conformance provenance: %w", err)
	}

	if err = suite.validate(); err != nil {
		return nil, err
	}

	return &suite, nil
}

type conformanceSuite struct {
	FormatVersion      int                 `json:"format_version"`
	ONNXSourceRevision string              `json:"onnx_source_revision"`
	ReferenceVersions  map[string]string   `json:"reference_versions"`
	OriginalLicense    string              `json:"original_license"`
	UpstreamLicense    string              `json:"upstream_license"`
	UpstreamAudit      conformanceArtifact `json:"upstream_audit"`
	Cases              []conformanceCase   `json:"cases"`
}

func (s *conformanceSuite) validate() (err error) {
	versions := map[string]string{"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3", "onnxruntime": "1.19.2"}
	if s.FormatVersion != 1 || s.ONNXSourceRevision != "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557" || !maps.Equal(s.ReferenceVersions, versions) || s.OriginalLicense == "" || s.UpstreamLicense == "" {
		return fmt.Errorf("conformance provenance header differs")
	}

	names := make(map[string]bool)
	for _, fixture := range s.Cases {
		if fixture.Name == "" || names[fixture.Name] || fixture.Origin == "" || fixture.License == "" {
			return fmt.Errorf("invalid/duplicate conformance case provenance %q", fixture.Name)
		}

		names[fixture.Name] = true
		var observation struct {
			Status string `json:"status"`
		}
		if err = json.Unmarshal(fixture.Observations["reference"], &observation); err != nil {
			return fmt.Errorf("case %q reference observation: %w", fixture.Name, err)
		}

		switch fixture.Status {
		case "agreement":
			if fixture.Reason != "" || fixture.IRVersion != 8 || !maps.Equal(fixture.Opsets, map[string]int64{"": 13}) || observation.Status != "returned" {
				return fmt.Errorf("case %q cannot claim profile agreement", fixture.Name)
			}
		case "extension":
			if fixture.Name != "m1_softmax_empty_axis" || fixture.Reason == "" || observation.Status != "rejected" {
				return fmt.Errorf("case %q unrecognized runtime extension", fixture.Name)
			}
		case "unsupported":
			if fixture.Reason == "" {
				return fmt.Errorf("case %q lacks an explicit exception reason", fixture.Name)
			}
		default:
			return fmt.Errorf("case %q unknown status %q", fixture.Name, fixture.Status)
		}
	}
	return nil
}

func TestConformanceSuite(t *testing.T) {
	suite, err := readConformanceSuite()
	if err != nil {
		t.Fatal(err)
	}

	counts := make(map[string]int)
	for _, fixture := range suite.Cases {
		counts[fixture.Status]++
		t.Run(fixture.Status+"/"+fixture.Name, func(t *testing.T) {
			outputs, runErr := fixture.run()
			if runErr != nil {
				t.Fatal(runErr)
			}

			// run closed both owners; retained public outputs must still compare.
			if fixture.Status != "unsupported" {
				if runErr = fixture.compareOutputs(outputs); runErr != nil {
					t.Fatal(runErr)
				}
			}
		})
	}
	if !maps.Equal(counts, map[string]int{"agreement": 86, "extension": 1, "unsupported": 9}) {
		t.Fatalf("case classification counts changed: %v", counts)
	}

	t.Logf("%d reference agreements, %d documented extensions, %d explicit rejections; no skips", counts["agreement"], counts["extension"], counts["unsupported"])
}

func TestConformanceSuiteDetectsNumericError(t *testing.T) {
	suite, err := readConformanceSuite()
	if err != nil {
		t.Fatal(err)
	}

	for _, fixture := range suite.Cases {
		if fixture.Name != "hand_add" {
			continue
		}

		outputs, runErr := fixture.run()
		if runErr != nil {
			t.Fatal(runErr)
		}

		// Deliberately inject a wrong result into a real executed fixture.
		outputs[0].Tensor = floatTensor(t, []int64{2}, []float32{4, 7})
		err = fixture.compareOutputs(outputs)
		if err == nil || !strings.Contains(err.Error(), `case "hand_add" inputs [left right]: output "y" flat[1] index [1]`) {
			t.Fatalf("numeric regression escaped conformance runner: %v", err)
		}

		t.Log(err)
		return
	}
	t.Fatal("hand-calculated regression fixture is missing")
}

func TestConformanceSuiteUpstreamAudit(t *testing.T) {
	suite, err := readConformanceSuite()
	if err != nil {
		t.Fatal(err)
	}

	data, err := suite.UpstreamAudit.read()
	if err != nil {
		t.Fatal(err)
	}

	var audit struct {
		SourceRevision string `json:"source_revision"`
		Total          int    `json:"total"`
		Compatible     int    `json:"ir8_opset13"`
		Models         []struct {
			Path      string           `json:"path"`
			SHA256    string           `json:"sha256"`
			IRVersion int64            `json:"ir_version"`
			Opsets    map[string]int64 `json:"opsets"`
		} `json:"models"`
	}
	if err = json.Unmarshal(data, &audit); err != nil {
		t.Fatal(err)
	}

	if audit.SourceRevision != suite.ONNXSourceRevision || audit.Total != 1422 || len(audit.Models) != audit.Total || audit.Compatible != 0 {
		t.Fatal("upstream audit header/count differs")
	}

	for _, model := range audit.Models {
		if model.IRVersion == 8 && maps.Equal(model.Opsets, map[string]int64{"": 13}) {
			t.Fatalf("unexpected compatible upstream model: %s", model.Path)
		}
	}
}
