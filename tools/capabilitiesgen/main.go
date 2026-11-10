// Command capabilitiesgen checks or updates the table from the runtime manifest.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"

	"github.com/itsmontoya/onnx"
)

func main() {
	write := flag.Bool("write", false, "update the generated capability table")
	flag.Parse()
	if err := generate(*write); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func generate(write bool) (err error) {
	const path = "CAPABILITIES.md"
	var data []byte
	if data, err = os.ReadFile(path); err != nil {
		return fmt.Errorf("read capability table: %w", err)
	}

	var (
		text                   = string(data)
		start                  = "<!-- Implemented capabilities start -->"
		end                    = "<!-- Implemented capabilities end -->"
		before, rest, hasStart = strings.Cut(text, start)
		_, after, hasEnd       = strings.Cut(rest, end)
	)

	if !hasStart || !hasEnd || strings.Count(text, start) != 1 || strings.Count(text, end) != 1 {
		return fmt.Errorf("capability table requires exactly one ordered marker pair")
	}

	generated := before + start + "\n\n" + capabilityTable() + "\n\n" + end + after
	if generated == text {
		fmt.Println("PASS: implemented table matches the runtime capability manifest")
		return nil
	}

	if !write {
		return fmt.Errorf("capability table differs; run go run ./tools/capabilitiesgen -write")
	}

	if err = os.WriteFile(path, []byte(generated), 0644); err != nil {
		return fmt.Errorf("write capability table: %w", err)
	}

	fmt.Println("Updated capability table from the runtime capability manifest")
	return nil
}

func capabilityTable() (table string) {
	var rows = []string{
		"| Operator (schema) | Tested imports | Types | Inputs → outputs | Attributes | Restrictions |",
		"| --- | --- | --- | --- | --- | --- |",
	}

	for _, entry := range onnx.Capabilities() {
		if entry.Status != "implemented" {
			continue
		}

		var attributes []string
		for _, name := range slices.Sorted(maps.Keys(entry.Attributes)) {
			attribute := entry.Attributes[name]
			value := name + ":" + attribute.Type
			if len(attribute.Default) != 0 {
				value += "=" + attributeDefault(attribute.Default)
			}

			attributes = append(attributes, "`"+value+"`")
		}

		attributeText := strings.Join(attributes, ", ")
		if len(attributes) == 0 {
			attributeText = "None"
		}

		rows = append(rows, fmt.Sprintf("| %s (%d) | %d–%d | %s | %s → %s | %s | %s |",
			entry.Operator, entry.SchemaVersion, entry.MinOpset, entry.MaxOpset, strings.Join(entry.DataTypes, ", "),
			arity(entry.Inputs), arity(entry.Outputs), attributeText,
			strings.Join(entry.Restrictions, "; ")))
	}

	return strings.Join(rows, "\n")
}

func arity(value onnx.CapabilityArity) (text string) {
	text = fmt.Sprint(value.Min)
	if value.Min != value.Max {
		text += fmt.Sprintf("–%d", value.Max)
	}

	if len(value.OptionalSlots) != 0 {
		text += fmt.Sprintf("; optional slots %v", value.OptionalSlots)
	}

	return text
}

func attributeDefault(value json.RawMessage) (text string) {
	var decoded string
	if err := json.Unmarshal(value, &decoded); err == nil {
		return decoded
	}

	return string(value)
}
