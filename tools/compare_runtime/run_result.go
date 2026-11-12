package main

type runResult struct {
	LoadNS      int64                   `json:"load_ns"`
	SessionNS   int64                   `json:"session_ns"`
	Outputs     map[string][]tensorData `json:"outputs"`
	LatenciesNS map[string][]int64      `json:"latencies_ns"`
}
