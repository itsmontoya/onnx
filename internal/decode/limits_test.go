package decode

import (
	"errors"
	"testing"
)

func TestLimits(t *testing.T) {
	tests := []struct {
		name  string
		input Limits
		want  Limits
		valid bool
	}{
		{"defaults", Limits{}, Limits{64 << 20, 64}, true},
		{"explicit", Limits{12, 3}, Limits{12, 3}, true},
		{"negative bytes", Limits{-1, 0}, Limits{}, false},
		{"negative recursion", Limits{0, -1}, Limits{}, false},
		{"overflow probe", Limits{int64(^uint(0) >> 1), 1}, Limits{}, false},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			result, err := testcase.input.normalized()
			if !testcase.valid {
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("got %v, want invalid argument", err)
				}

				return
			}

			if err != nil || result != testcase.want {
				t.Fatalf("got %+v, %v; want %+v", result, err, testcase.want)
			}
		})
	}
}
