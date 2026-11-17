//go:build ignore

// Generates tests/ollama-engine/gotmpl_oracle.json: real ollama template.Execute output for a template x conversation grid.
// Run from an ollama checkout:  cp gen_gotmpl_oracle.go <ollama>/cmd/oraclegen/main.go && (cd <ollama> && go run ./cmd/oraclegen > gotmpl_oracle.json)
// The expected strings are FACTS produced by ollama's own engine; the C++ test (tests/test-ollama-gotmpl.cpp) renders the
// translated Jinja with the fork's engine and compares byte for byte. No ollama code is copied into the fork.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

type caseOut struct {
	Name     string        `json:"name"`
	Template string        `json:"template"`
	Messages []api.Message `json:"messages"`
	Tools    api.Tools     `json:"tools,omitempty"`
	Expected string        `json:"expected"`
	Error    string        `json:"error,omitempty"`
}

func main() {
	convs := map[string][]api.Message{
		"u":     {{Role: "user", Content: "Hello, how are you?"}},
		"uau":   {{Role: "user", Content: "Hello, how are you?"}, {Role: "assistant", Content: "I'm doing great. How can I help you today?"}, {Role: "user", Content: "I'd like to show off how chat templating works!"}},
		"suau":  {{Role: "system", Content: "You are a helpful assistant."}, {Role: "user", Content: "Hello, how are you?"}, {Role: "assistant", Content: "I'm doing great. How can I help you today?"}, {Role: "user", Content: "I'd like to show off how chat templating works!"}},
		"uu":    {{Role: "user", Content: "first"}, {Role: "user", Content: "second"}},
		"ssua":  {{Role: "system", Content: "sys one"}, {Role: "system", Content: "sys two"}, {Role: "user", Content: "q"}, {Role: "assistant", Content: "a"}},
		"ua":    {{Role: "user", Content: "q"}, {Role: "assistant", Content: "a"}},
	}
	extra := map[string]string{
		"legacy-mistral":   "[INST] {{ if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}[/INST] {{ .Response }}",
		"legacy-noresp":    "[INST] {{ if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}[/INST] ",
		"legacy-llama2":    "{{ if .System }}<<SYS>>{{ .System }}<</SYS>>\n{{ end }}{{ .Prompt }}",
		"msgs-vars":        "{{- $n := 0 }}{{- range .Messages }}{{- $n = len .Content }}[{{ .Role }}:{{ $n }}]{{ end }}",
		"msgs-index":       "{{- range $i, $m := .Messages }}{{ if $i }},{{ end }}{{ $i }}={{ $m.Role }}{{ end }}",
		"msgs-with":        "{{- with .System }}S:{{ . }}\n{{ end }}{{ range .Messages }}{{ .Role }};{{ end }}",
		"msgs-and-or-not":  "{{- range .Messages }}{{ if and (ne .Role \"system\") (not (eq .Content \"\")) }}{{ .Role }}{{ else if or (eq .Role \"system\") false }}SYS{{ end }}.{{ end }}",
		"msgs-eq-multi":    "{{- range .Messages }}{{ if eq .Role \"user\" \"assistant\" }}U/A{{ else }}other{{ end }} {{ end }}",
		"msgs-printf":      "{{- range .Messages }}{{ printf \"<%s|%d>\" .Role 7 }}{{ end }}",
		"msgs-literal":     "{{ \"{{\" }}x{{ \"}}\" }} {{ range .Messages }}{{ .Content }}{{ end }}",
		"msgs-comment":     "a{{/* note */}}b{{- /* trimmed */ -}}  c{{ range .Messages }}{{ .Role }}{{ end }}",
		"msgs-break":       "{{- range .Messages }}{{ if eq .Role \"assistant\" }}{{ break }}{{ end }}{{ .Role }},{{ end }}",
		"msgs-slice-len":   "{{- range $i, $_ := .Messages }}{{ $last := eq (len (slice $.Messages $i)) 1 }}{{ if $last }}LAST{{ else }}.{{ end }}{{ end }}",
		"msgs-json":        "{{- range .Messages }}{{ json .Content }}{{ end }}",
		"msgs-indent":      "A\n    {{ if .System }}S{{ end }}\n    {{ range .Messages }}[{{ .Role }}]{{ end }}\n  {{ if .System }}tail{{ end }}\nB\n",
	}
	tools := api.Tools{{
		Type: "function",
		Function: api.ToolFunction{
			Name: "get_weather", Description: "Get the weather",
			Parameters: api.ToolFunctionParameters{Type: "object", Required: []string{"location"}, Properties: func() *api.ToolPropertiesMap {
				m := api.NewToolPropertiesMap()
				m.Set("zeta", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "last alphabetically, first inserted"})
				m.Set("alpha", api.ToolProperty{Type: api.PropertyType{"number"}, Description: "first alphabetically"})
				return m
			}()},
		},
	}}
	toolTmpl := map[string]string{
		"tools-range":      "{{- range .Tools }}{{ .Function.Name }}({{ range $name, $p := .Function.Parameters.Properties }}{{ $name }}:{{ $p.Type }},{{ end }}){{ end }}{{ range .Messages }}|{{ .Role }}{{ end }}",
		"tools-if":         "{{- if .Tools }}HAS{{ else }}NONE{{ end }}{{ range .Messages }}|{{ .Role }}{{ end }}",
		"tools-calls":      "{{- range .Messages }}{{ .Role }}{{ range .ToolCalls }}<{{ .Function.Name }}>{{ end }};{{ end }}",
	}
	toolConvs := map[string][]api.Message{
		"tc": {{Role: "user", Content: "weather?"}, {Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "get_weather"}}}}, {Role: "tool", Content: "sunny", ToolName: "get_weather"}},
	}

	var out []caseOut
	run := func(name, tmpl string, msgs []api.Message, tl api.Tools) {
		c := caseOut{Name: name, Template: tmpl, Messages: msgs, Tools: tl}
		t, err := template.Parse(tmpl)
		if err != nil {
			c.Error = err.Error()
		} else {
			var b bytes.Buffer
			cp := append([]api.Message(nil), msgs...)
			if err := t.Execute(&b, template.Values{Messages: cp, Tools: tl}); err != nil {
				c.Error = err.Error()
			} else {
				c.Expected = b.String()
			}
		}
		out = append(out, c)
	}
	files, _ := filepath.Glob("template/*.gotmpl")
	sort.Strings(files)
	keys := func(m map[string][]api.Message) []string { var k []string; for x := range m { k = append(k, x) }; sort.Strings(k); return k }
	for _, f := range files {
		b, _ := os.ReadFile(f)
		for _, k := range keys(convs) {
			run(filepath.Base(f)+"/"+k, string(b), convs[k], nil)
		}
	}
	var names []string
	for n := range extra { names = append(names, n) }
	sort.Strings(names)
	for _, n := range names {
		for _, k := range keys(convs) {
			run(n+"/"+k, extra[n], convs[k], nil)
		}
	}
	names = nil
	for n := range toolTmpl { names = append(names, n) }
	sort.Strings(names)
	for _, n := range names {
		for _, k := range keys(toolConvs) {
			run(n+"/"+k, toolTmpl[n], toolConvs[k], tools)
		}
		run(n+"/u", toolTmpl[n], convs["u"], tools)
	}
	// command-r uses .Tools with the real shape
	b, _ := os.ReadFile("template/command-r.gotmpl")
	run("command-r.gotmpl/tools", string(b), convs["uau"], tools)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", " ")
	if err := enc.Encode(out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
