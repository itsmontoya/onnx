package decode

import (
	"bytes"
	"context"
	"os"
	"testing"

	"google.golang.org/protobuf/proto"
)

func FuzzProtobufEntrypoints(f *testing.F) {
	for _, path := range []string{
		"../../testdata/profile/identity.onnx",
		"../../testdata/conformance/values/hand_exact_int64/input_exact.pb",
	} {
		data, err := os.ReadFile(path)
		if err != nil {
			f.Fatal(err)
		}

		f.Add(data)
	}
	// Literal malformed encodings also exercised by TestMalformedWire.
	for _, data := range [][]byte{{}, {0x08, 0x80}, {0x3a, 0x02, 0x08}, {0x0f}, {0x0c}, {0x22, 0x01, 0x00}, {0x0b, 0x0c}} {
		f.Add(data)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 16<<10 {
			t.Skip()
		}

		limits := Limits{MaxBytes: 16 << 10, MaxRecursion: 16}
		parsedModel, parseModelErr := ParseModel(context.Background(), data, limits)
		readModel, readModelErr := ReadModel(context.Background(), bytes.NewReader(data), limits)
		if (parseModelErr == nil) != (readModelErr == nil) || !proto.Equal(parsedModel, readModel) {
			t.Fatalf("model byte/reader disagreement: %v, %v", parseModelErr, readModelErr)
		}

		parsedTensor, parseTensorErr := ParseTensor(context.Background(), data, limits)
		readTensor, readTensorErr := ReadTensor(context.Background(), bytes.NewReader(data), limits)
		if (parseTensorErr == nil) != (readTensorErr == nil) || !proto.Equal(parsedTensor, readTensor) {
			t.Fatalf("tensor byte/reader disagreement: %v, %v", parseTensorErr, readTensorErr)
		}
	})
}
