package onnx

import (
	"fmt"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/operators"
)

// Limits bounds logical payload and tensor reservations, excluding Go allocation
// overhead and outputs retained from earlier runs. Retained Session destination
// payload is included in MaxLiveBytes in addition to invocation reservations.
// Zero fields select defaults; negative or host-unrepresentable limits fail
// before any I/O.
type Limits struct {
	MaxModelBytes       int64
	MaxLoadBytes        int64
	MaxInitializerBytes int64
	MaxTensorBytes      int64
	MaxLiveBytes        int64
	MaxWorkspaceBytes   int64
	MaxNodes            int
	MaxValues           int
	MaxAttributes       int
	MaxNameBytes        int
	MaxRank             int
	MaxDimension        int64
	MaxRecursion        int
	MaxSessions         int
}

func (l Limits) normalized() (out Limits, err error) {
	var (
		bytes         = []*int64{&l.MaxModelBytes, &l.MaxLoadBytes, &l.MaxInitializerBytes, &l.MaxTensorBytes, &l.MaxLiveBytes, &l.MaxWorkspaceBytes, &l.MaxDimension}
		byteDefaults  = []int64{64 << 20, 1 << 30, 256 << 20, 64 << 20, 512 << 20, 256 << 20, 1<<31 - 1}
		counts        = []*int{&l.MaxNodes, &l.MaxValues, &l.MaxAttributes, &l.MaxNameBytes, &l.MaxRank, &l.MaxRecursion, &l.MaxSessions}
		countDefaults = []int{100000, 200000, 400000, 4096, 32, 64, 16}
		hostMax       = int64(^uint(0) >> 1)
	)

	for index, field := range bytes {
		if *field < 0 || *field > hostMax {
			return out, fmt.Errorf("%w: negative or host-unrepresentable byte/dimension limit", ErrInvalidArgument)
		}

		if *field == 0 {
			*field = byteDefaults[index]
		}
	}
	for index, field := range counts {
		if *field < 0 {
			return out, fmt.Errorf("%w: negative count limit", ErrInvalidArgument)
		}

		if *field == 0 {
			*field = countDefaults[index]
		}
	}

	if l.MaxModelBytes >= hostMax || int64(l.MaxRank) > hostMax/16 {
		return out, fmt.Errorf("%w: model limit plus one or rank payload must fit host arithmetic", ErrInvalidArgument)
	}

	return l, nil
}

func (l Limits) wire() (out decode.Limits) {
	return decode.Limits{MaxBytes: l.MaxModelBytes, MaxRecursion: l.MaxRecursion}
}

func (l Limits) graph() (out graph.Limits) {
	return graph.Limits{
		Tensor: decode.TensorLimits{
			MaxLoadBytes: l.MaxLoadBytes, MaxInitializerBytes: l.MaxInitializerBytes,
			MaxTensorBytes: l.MaxTensorBytes, MaxRank: l.MaxRank,
			MaxDimension: l.MaxDimension, MaxRecursion: l.MaxRecursion,
		},
		MaxNodes: l.MaxNodes, MaxValues: l.MaxValues,
		MaxAttributes: l.MaxAttributes, MaxNameBytes: l.MaxNameBytes,
	}
}

func (l Limits) compilation() (out operators.Limits) {
	return operators.Limits{MaxLoadBytes: l.MaxLoadBytes, MaxTensorBytes: l.MaxTensorBytes, MaxRank: l.MaxRank, MaxDimension: l.MaxDimension}
}
