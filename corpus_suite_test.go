package onnx

import (
	"encoding/json"
	"os"
	"testing"
)

type corpusSuite struct {
	Source         string              `json:"source_revision"`
	ArchiveSHA256  string              `json:"archive_sha256"`
	SourceExport   conformanceArtifact `json:"source_export"`
	UpstreamModels int                 `json:"upstream_model_count"`
	Exact13        int                 `json:"upstream_exact_ir8_opset13"`
	Exact17        int                 `json:"upstream_exact_ir8_opset17"`
	Models         []corpusModel       `json:"models"`
	Before         struct {
		Findings int `json:"findings"`
		Enabled  int `json:"enabled"`
	} `json:"before_rt19"`
	After struct {
		Findings int `json:"findings"`
		Enabled  int `json:"enabled"`
	} `json:"after_rt19"`
}

func TestAuditedCorpus(t *testing.T) {
	data, err := os.ReadFile("testdata/corpus/index.json")
	if err != nil {
		t.Fatal(err)
	}

	var suite corpusSuite
	if err = json.Unmarshal(data, &suite); err != nil {
		t.Fatal(err)
	}

	if suite.Source != "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557" || suite.ArchiveSHA256 != "d32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5" || suite.UpstreamModels != 1422 || suite.Exact13 != 0 || suite.Exact17 != 65 || len(suite.Models) != 33 {
		t.Fatal("corpus source/count differs")
	}

	if _, err = suite.SourceExport.read(); err != nil {
		t.Fatal(err)
	}

	var before, after, enabled int
	names := make(map[string]bool)
	for _, model := range suite.Models {
		if model.Name == "" || names[model.Name] {
			t.Fatal("missing/duplicate corpus model name")
		}

		names[model.Name] = true
		before += model.Before.Count
		after += model.After.Count
		if model.Status == "enabled" {
			enabled++
		}

		t.Run(model.Name, func(t *testing.T) { model.check(t) })
	}
	if before != 63 || after != 44 || enabled != 19 || suite.Before.Findings != before || suite.After.Findings != after || suite.Before.Enabled != 0 || suite.After.Enabled != enabled {
		t.Fatal("deterministic corpus finding totals differ")
	}
}
