// Command onnxgen reproducibly generates and splits the pinned ONNX Go bindings.
package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const protobufVersion = "v1.36.12"

func main() {
	if err := generate(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func generate() (err error) {
	var (
		temporary string
		input     []byte
		files     map[string][]byte
	)

	if err = checkTools(); err != nil {
		return err
	}

	if err = checkSource(); err != nil {
		return err
	}

	if temporary, err = os.MkdirTemp("", "onnxgen-"); err != nil {
		return fmt.Errorf("create generation directory: %w", err)
	}

	defer func() {
		var cleanupErr error
		if cleanupErr = os.RemoveAll(temporary); cleanupErr != nil && err == nil {
			err = fmt.Errorf("remove generation directory: %w", cleanupErr)
		}
	}()

	plugin := filepath.Join(temporary, "protoc-gen-go")
	if err = run("go", "build", "-o", plugin, "google.golang.org/protobuf/cmd/protoc-gen-go"); err != nil {
		return err
	}

	if err = run("protoc", "--plugin=protoc-gen-go="+plugin,
		"--proto_path=third_party/onnx", "--go_out="+temporary,
		"--go_opt=paths=source_relative",
		"--go_opt=Monnx.proto=github.com/itsmontoya/onnx/internal/onnxpb", "onnx.proto"); err != nil {
		return err
	}

	if input, err = os.ReadFile(filepath.Join(temporary, "onnx.pb.go")); err != nil {
		return fmt.Errorf("read generated bindings: %w", err)
	}

	if files, err = split(input); err != nil {
		return err
	}

	return writeFiles(files)
}

func checkTools() (err error) {
	var output []byte
	if output, err = exec.Command("protoc", "--version").Output(); err != nil {
		return fmt.Errorf("inspect protoc: %w", err)
	}

	if strings.TrimSpace(string(output)) != "libprotoc 33.2" {
		return fmt.Errorf("generation requires protoc 33.2; got %q", strings.TrimSpace(string(output)))
	}

	if output, err = exec.Command("go", "list", "-m", "-f", "{{.Version}}", "google.golang.org/protobuf").Output(); err != nil {
		return fmt.Errorf("inspect Go protobuf version: %w", err)
	}

	if strings.TrimSpace(string(output)) != protobufVersion {
		return fmt.Errorf("generation requires google.golang.org/protobuf %s", protobufVersion)
	}

	return nil
}

func checkSource() (err error) {
	var source []byte
	if source, err = os.ReadFile("third_party/onnx/onnx.proto"); err != nil {
		return fmt.Errorf("read pinned schema: %w", err)
	}

	const expected = "1023c7f39a568f309fdd0b445a81a7a0d90f74046e5967c2e817d4e4e06a5f1b"
	if actual := fmt.Sprintf("%x", sha256.Sum256(source)); actual != expected {
		return fmt.Errorf("pinned ONNX schema hash: got %s, want %s", actual, expected)
	}

	return nil
}

func run(name string, arguments ...string) (err error) {
	command := exec.Command(name, arguments...)
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err = command.Run(); err != nil {
		return fmt.Errorf("run %s: %w", name, err)
	}

	return nil
}

func split(source []byte) (files map[string][]byte, err error) {
	var (
		positions token.FileSet
		file      *ast.File
		groups    = make(map[string][]ast.Decl)
		types     []string
	)

	if file, err = parser.ParseFile(&positions, "onnx.pb.go", source, parser.ParseComments); err != nil {
		return nil, fmt.Errorf("parse generated bindings: %w", err)
	}

	for _, declaration := range file.Decls {
		if group, ok := declaration.(*ast.GenDecl); ok && group.Tok == token.TYPE {
			for _, specification := range group.Specs {
				types = append(types, specification.(*ast.TypeSpec).Name.Name)
			}
		}
	}

	// Longest prefixes distinguish a nested enum from its parent message.
	sort.Slice(types, func(left, right int) bool { return len(types[left]) > len(types[right]) })
	for _, declaration := range file.Decls {
		if group, ok := declaration.(*ast.GenDecl); ok && group.Tok == token.IMPORT {
			continue
		}

		owner := declarationOwner(declaration, types)
		groups[owner] = append(groups[owner], declaration)
	}

	files = make(map[string][]byte, len(groups))
	for owner, declarations := range groups {
		var generated []byte
		if generated, err = render(source, &positions, file, declarations); err != nil {
			return nil, err
		}

		files[fileName(owner)] = generated
	}

	return files, nil
}

func declarationOwner(declaration ast.Decl, types []string) (owner string) {
	switch value := declaration.(type) {
	case *ast.FuncDecl:
		if value.Recv != nil {
			return receiverName(value.Recv.List[0].Type)
		}
	case *ast.GenDecl:
		if value.Tok == token.TYPE {
			return value.Specs[0].(*ast.TypeSpec).Name.Name
		}

		for _, specification := range value.Specs {
			variable := specification.(*ast.ValueSpec)
			for _, name := range variable.Names {
				for _, typeName := range types {
					if strings.HasPrefix(name.Name, typeName+"_") {
						return typeName
					}
				}
			}
		}
	}

	return "schema"
}

func receiverName(expression ast.Expr) (name string) {
	if pointer, ok := expression.(*ast.StarExpr); ok {
		return receiverName(pointer.X)
	}

	return expression.(*ast.Ident).Name
}

func render(source []byte, positions *token.FileSet, file *ast.File, declarations []ast.Decl) (output []byte, err error) {
	var (
		buffer       bytes.Buffer
		dependencies bytes.Buffer
		used         = make(map[string]bool)
	)

	buffer.Write(source[:positions.Position(file.Package).Offset])
	buffer.WriteString("// Split deterministically by tools/onnxgen; do not edit individual files.\n\npackage onnxpb\n\n")
	for _, declaration := range declarations {
		ast.Inspect(declaration, func(node ast.Node) bool {
			if selector, ok := node.(*ast.SelectorExpr); ok {
				if identifier, valid := selector.X.(*ast.Ident); valid {
					used[identifier.Name] = true
				}
			}

			return true
		})
	}

	for _, dependency := range file.Imports {
		if used[dependency.Name.Name] {
			fmt.Fprintf(&dependencies, "%s %s\n", dependency.Name.Name, dependency.Path.Value)
		}
	}

	if dependencies.Len() > 0 {
		buffer.WriteString("import (\n")
		buffer.Write(dependencies.Bytes())
		buffer.WriteString(")\n\n")
	}

	for _, declaration := range declarations {
		start := declaration.Pos()
		switch value := declaration.(type) {
		case *ast.FuncDecl:
			if value.Doc != nil {
				start = value.Doc.Pos()
			}
		case *ast.GenDecl:
			if value.Doc != nil {
				start = value.Doc.Pos()
			}
		}

		buffer.Write(source[positions.Position(start).Offset:positions.Position(declaration.End()).Offset])
		buffer.WriteString("\n\n")
	}

	if output, err = format.Source(buffer.Bytes()); err != nil {
		return nil, fmt.Errorf("format split bindings: %w", err)
	}

	return output, nil
}

func fileName(owner string) (name string) {
	boundaries := regexp.MustCompile(`([a-z0-9])([A-Z])`)
	return strings.ToLower(boundaries.ReplaceAllString(owner, "${1}_${2}")) + ".pb.go"
}

func writeFiles(files map[string][]byte) (err error) {
	const directory = "internal/onnxpb"
	var existing []string
	if err = os.MkdirAll(directory, 0755); err != nil {
		return fmt.Errorf("create bindings directory: %w", err)
	}

	if existing, err = filepath.Glob(filepath.Join(directory, "*.pb.go")); err != nil {
		return fmt.Errorf("list existing bindings: %w", err)
	}

	for _, path := range existing {
		if _, retained := files[filepath.Base(path)]; !retained {
			if err = os.Remove(path); err != nil {
				return fmt.Errorf("remove stale bindings %q: %w", path, err)
			}
		}
	}

	for name, contents := range files {
		path := filepath.Join(directory, name)
		if err = os.WriteFile(path, contents, 0644); err != nil {
			return fmt.Errorf("write bindings %q: %w", path, err)
		}
	}

	return nil
}
