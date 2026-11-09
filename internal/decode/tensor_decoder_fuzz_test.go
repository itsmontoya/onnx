package decode

import (
	"context"
	"os"
	"testing"
)

func FuzzTensorDecode(f *testing.F) {
	for _, path := range []string{
		"../../testdata/conformance/values/hand_exact_int64/input_exact.pb",
		"../../testdata/conformance/values/hand_empty_int64/input_empty.pb",
		"../../testdata/conformance/values/hand_scalar/input_scalar.pb",
		"../../testdata/conformance/values/hand_special_identity/input_special.pb",
	} {
		data, err := os.ReadFile(path)
		if err != nil {
			f.Fatal(err)
		}

		f.Add(data)
	}
	for _, data := range [][]byte{{}, {0x08, 0x80}, {0x10, 1, 0x4a, 1, 0}, {0x08, 0, 0x10, 7}, {0x08, 2, 0x10, 1, 0x22, 0}} {
		f.Add(data)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 16<<10 {
			t.Skip()
		}

		ctx := context.Background()
		wire, err := ParseTensor(ctx, data, Limits{MaxBytes: 16 << 10, MaxRecursion: 16})
		if err != nil {
			return
		}

		limits := TensorLimits{
			MaxLoadBytes: 128 << 10, MaxInitializerBytes: 32 << 10,
			MaxTensorBytes: 16 << 10, MaxRank: 8, MaxDimension: 1024, MaxRecursion: 16,
		}
		decoder, err := NewTensorDecoder(ctx, wire, int64(len(data)), limits)
		if err != nil {
			return
		}

		beforeLoad, beforeOwned := decoder.Usage()
		value, decodeErr := decoder.Decode(ctx, wire)
		load, owned := decoder.Usage()
		if decodeErr != nil {
			if value != nil || load != beforeLoad || owned != beforeOwned {
				t.Fatalf("failed conversion published storage or reservation: %v", decodeErr)
			}

			return
		}

		if value == nil || value.Validate() != nil || value.ByteSize() > limits.MaxTensorBytes || load > limits.MaxLoadBytes || owned > limits.MaxInitializerBytes {
			t.Fatal("successful conversion exceeded finite limits or returned invalid storage")
		}
	})
}
