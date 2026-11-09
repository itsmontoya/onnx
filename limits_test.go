package onnx

import (
	"errors"
	"reflect"
	"testing"
)

func TestLimitsDefaultsAndInvalidFields(t *testing.T) {
	var zero Limits
	defaults, err := zero.normalized()
	want := Limits{
		MaxModelBytes: 64 << 20, MaxLoadBytes: 1 << 30,
		MaxInitializerBytes: 256 << 20, MaxTensorBytes: 64 << 20,
		MaxLiveBytes: 512 << 20, MaxWorkspaceBytes: 256 << 20,
		MaxNodes: 100000, MaxValues: 200000, MaxAttributes: 400000,
		MaxNameBytes: 4096, MaxRank: 32, MaxDimension: 1<<31 - 1,
		MaxRecursion: 64, MaxSessions: 16,
	}
	if err != nil || defaults != want || zero != (Limits{}) {
		t.Fatalf("normalized defaults: %+v, %v", defaults, err)
	}

	// Every exported limit participates in validation; reflection here only
	// enumerates fields, with no test-specific dependency on implementation.
	for index := 0; index < reflect.TypeFor[Limits]().NumField(); index++ {
		field := reflect.TypeFor[Limits]().Field(index)
		t.Run(field.Name, func(t *testing.T) {
			var limits Limits
			reflect.ValueOf(&limits).Elem().Field(index).SetInt(-1)
			if _, normalizeErr := limits.normalized(); !errors.Is(normalizeErr, ErrInvalidArgument) {
				t.Fatalf("negative %s accepted: %v", field.Name, normalizeErr)
			}

			reflect.ValueOf(&limits).Elem().Field(index).SetInt(1)
			out, normalizeErr := limits.normalized()
			if normalizeErr != nil || reflect.ValueOf(out).Field(index).Int() != 1 {
				t.Fatalf("positive %s was not preserved: %+v, %v", field.Name, out, normalizeErr)
			}
		})
	}

	hostMax := int64(^uint(0) >> 1)
	for _, limits := range []Limits{{MaxModelBytes: hostMax}, {MaxRank: int(hostMax/16) + 1}} {
		if _, err = limits.normalized(); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("host arithmetic limit accepted: %+v, %v", limits, err)
		}
	}
}
