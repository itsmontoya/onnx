package onnx

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path"
	"strings"
)

type conformanceArtifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

func (a conformanceArtifact) read() (data []byte, err error) {
	if a.Path == "" || path.IsAbs(a.Path) || path.Clean(a.Path) != a.Path || strings.HasPrefix(a.Path, "../") {
		return nil, fmt.Errorf("invalid fixture path %q", a.Path)
	}

	if data, err = os.ReadFile("testdata/" + a.Path); err != nil {
		return nil, fmt.Errorf("read fixture %q: %w", a.Path, err)
	}

	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != a.SHA256 {
		return nil, fmt.Errorf("fixture %q SHA-256 mismatch", a.Path)
	}

	return data, nil
}
