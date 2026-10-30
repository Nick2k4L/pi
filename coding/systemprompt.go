package coding

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/sky-valley/pi/ai"
	"github.com/sky-valley/pi/internal/jstext"
)

// System prompt construction, ported from pi
// packages/coding-agent/src/core/system-prompt.ts (upstream 9e05370b2 and
// e4c75a732). The prompt is a set of ordered, independently replaceable
// sections that become the transcript's SystemMessage.Sections, or, when
// forced, one opaque text that becomes its Content; BuildSystemPrompt renders
// either the way the transcript replays it.

// ContextFile is a project context file injected into the system prompt.
type ContextFile struct {
	Path    string
	Content string
}

// BuildSystemPromptOptions configures BuildSystemPromptSections and
// BuildSystemPrompt (pi BuildSystemPromptOptions).
type BuildSystemPromptOptions struct {
	// CustomPrompt replaces the default prompt prefix — the preamble, tools,
	// rules and docs sections — with its text as the preamble. Empty keeps the
	// default prefix.
	CustomPrompt string
	// ForceSystemPrompt, when non-nil, is an exact full prompt replacement:
	// BuildSystemPromptState and BuildSystemPrompt take its text verbatim as the
	// whole prompt, with no sections, even when empty; BuildSystemPromptSections
	// ignores it. pi sets it from a before_agent_start handler.
	ForceSystemPrompt *string
	// SelectedTools are the active tool names. Nil means pi's default
	// [read, bash, edit, write]; an empty, non-nil slice means no tools.
	SelectedTools []string
	// HiddenTools are selected tools whose declarations requests leave out
	// (pi's prepareLoadout hidden declarations). They are reachable only
	// through another tool, so the tool list and the rules leave them out too;
	// a hidden read or bash still offers skills, with a hint that names no
	// tool (upstream c30840c2e).
	HiddenTools []string
	// ToolSnippets are optional one-line tool snippets keyed by tool name. A
	// selected tool appears in the tools section only when it has one.
	ToolSnippets map[string]string
	// ToolGuidelines are the guideline bullets each tool contributes, keyed by
	// tool name. Only selected tools' guidelines reach the rules section.
	ToolGuidelines map[string][]string
	// PromptGuidelines are additional guideline bullets appended to the rules.
	PromptGuidelines []string
	// AppendSystemPrompt is text from user configuration placed before project
	// context, skills and cwd (the addendum section).
	AppendSystemPrompt string
	// Sections are additional prompt sections, each wrapped in a tag of its
	// name. A name must start with a lowercase letter followed by lowercase
	// letters, digits, "_" or "-", and "preamble" is reserved. A section named
	// like a built-in one replaces its content in place; an empty or nil value
	// adds nothing.
	Sections ai.SystemSections
	// Cwd is the working directory; backslashes render as forward slashes.
	Cwd string
	// ContextFiles are pre-loaded project context files.
	ContextFiles []ContextFile
	// Skills are pre-loaded skills, offered when read or bash is selected.
	Skills []Skill
	// ReadmePath/DocsPath/ExamplesPath are the absolute pi documentation paths
	// referenced by the docs section. Empty values fall back to
	// ReadmePath()/DocsPath()/ExamplesPath().
	ReadmePath   string
	DocsPath     string
	ExamplesPath string
	// VerifyDocPaths, when true, makes the docs section advertise only what
	// actually exists on disk: the Main documentation line only when the README
	// exists, the Additional docs / Examples lines only when their directories
	// exist, and each "When asked about" topic only when at least one of its
	// referenced files (or directories) does. When nothing exists, the docs
	// section is omitted entirely. Upstream pi always advertises the full
	// documentation set regardless of the filesystem; the coding session
	// enables this so the prompt stays truthful when the binary runs from a
	// checkout that does not carry pi's docs.
	VerifyDocPaths bool
}

// defaultPromptPreamble is the default prompt's untagged opening section.
const defaultPromptPreamble = "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files."

// systemPromptSectionName is pi's SYSTEM_PROMPT_SECTION_NAME.
var systemPromptSectionName = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)

// NormalizeBuildSystemPromptOptions returns opts in pi's normalized,
// collection-complete shape (normalizeBuildSystemPromptOptions): SelectedTools
// defaults to [read, bash, edit, write] only when nil, HiddenTools to none,
// every collection is
// non-nil, and every collection is a copy, so mutating the result cannot reach
// the caller's inputs or the other way round. Go has one options type for both
// shapes; the normalized one is the result of this function.
func NormalizeBuildSystemPromptOptions(opts BuildSystemPromptOptions) BuildSystemPromptOptions {
	out := opts
	if opts.SelectedTools == nil {
		out.SelectedTools = []string{"read", "bash", "edit", "write"}
	} else {
		out.SelectedTools = slices.Clone(opts.SelectedTools)
	}
	out.HiddenTools = append([]string{}, opts.HiddenTools...)
	out.ToolSnippets = maps.Clone(opts.ToolSnippets)
	if out.ToolSnippets == nil {
		out.ToolSnippets = map[string]string{}
	}
	out.ToolGuidelines = make(map[string][]string, len(opts.ToolGuidelines))
	for name, guidelines := range opts.ToolGuidelines {
		out.ToolGuidelines[name] = slices.Clone(guidelines)
	}
	out.PromptGuidelines = append([]string{}, opts.PromptGuidelines...)
	// A spread copy is a fresh object in the source's own key order.
	out.Sections = ai.SystemSections{}
	for _, section := range opts.Sections.Entries() {
		value := section.Value
		if value != nil {
			content := *value
			value = &content
		}
		out.Sections.Set(section.Name, value)
	}
	out.ContextFiles = append([]ContextFile{}, opts.ContextFiles...)
	out.Skills = append([]Skill{}, opts.Skills...)
	return out
}

// renderProjectContext is pi's renderProjectContext.
func renderProjectContext(contextFiles []ContextFile) string {
	parts := []string{"Project-specific instructions and guidelines:"}
	for _, file := range contextFiles {
		// pi interpolates the raw path into the attribute — no quoting or
		// escaping, byte for byte.
		parts = append(parts, `<project_instructions path="`+file.Path+`">`+"\n"+file.Content+"\n</project_instructions>")
	}
	return strings.Join(parts, "\n\n")
}

// buildRules is pi's buildRules: the file-exploration rule, each selected
// tool's guidelines in selection order, the prompt guidelines, then the two
// fixed rules — each trimmed as JS trims, empties dropped, first occurrence
// kept.
func buildRules(selectedTools []string, toolGuidelines map[string][]string, promptGuidelines []string) string {
	var rules []string
	seen := map[string]bool{}
	addRule := func(rule string) {
		normalized := jstext.Trim(rule)
		if normalized == "" || seen[normalized] {
			return
		}
		seen[normalized] = true
		rules = append(rules, normalized)
	}

	has := func(name string) bool { return slices.Contains(selectedTools, name) }
	// With no dedicated exploration tool the rule names whichever shells are
	// active; the bash-only wording predates powershell (upstream 80e62761f).
	hasBash, hasPowerShell := has("bash"), has("powershell")
	if (hasBash || hasPowerShell) && !has("grep") && !has("find") && !has("ls") {
		switch {
		case hasBash && hasPowerShell:
			addRule("Use bash or PowerShell for file operations like listing, searching, and finding files")
		case hasPowerShell:
			addRule("Use PowerShell for file operations like listing, searching, and finding files")
		default:
			addRule("Use bash for file operations like ls, rg, find")
		}
	}

	for _, name := range selectedTools {
		for _, rule := range toolGuidelines[name] {
			addRule(rule)
		}
	}
	for _, rule := range promptGuidelines {
		addRule(rule)
	}
	addRule("Be concise in your responses")
	addRule("Show file paths clearly when working with files")

	lines := make([]string, len(rules))
	for i, rule := range rules {
		lines[i] = "- " + rule
	}
	return strings.Join(lines, "\n")
}

// invalidSectionNameError is pi's "Invalid system prompt section name" throw,
// with what a valid name looks like.
func invalidSectionNameError(name string) error {
	return fmt.Errorf(`Invalid system prompt section name: %s (a section name starts with a lowercase letter followed by lowercase letters, digits, "_" or "-", and "preamble" is reserved; rename it, e.g. "plan_mode")`, name)
}

// BuildSystemPromptSections builds the ordered, independently replaceable
// sections of the structured system prompt (pi buildSystemPromptSections): the
// untagged preamble, then for the default prompt tools, rules and docs, then
// addendum, project_context, skills and cwd, then the custom Sections. Every
// section but the preamble is wrapped in a tag of its name, so the model can
// match later updates to it. These become SystemMessage.Sections in the
// transcript. ForceSystemPrompt plays no part: a forced prompt has no sections
// (BuildSystemPromptState).
//
// It fails, building nothing, when a custom section name is invalid; the first
// invalid name in JS key order is reported.
func BuildSystemPromptSections(input BuildSystemPromptOptions) (ai.SystemSections, error) {
	opts := NormalizeBuildSystemPromptOptions(input)
	customSections := opts.Sections.Entries()
	for _, section := range customSections {
		if !systemPromptSectionName.MatchString(section.Name) || section.Name == "preamble" {
			return nil, invalidSectionNameError(section.Name)
		}
	}

	// The declared tools are the selected ones whose declarations reach the
	// model; only they appear in the tool list and contribute rules.
	declaredTools := slices.DeleteFunc(slices.Clone(opts.SelectedTools), func(name string) bool {
		return slices.Contains(opts.HiddenTools, name)
	})
	// promptSections is pi's plain object: assigning an existing name keeps its
	// slot, which is how a custom section replaces a built-in one in place.
	promptSections := ai.SystemSections{}
	set := func(name, content string) { promptSections.Set(name, &content) }
	if opts.CustomPrompt != "" {
		set("preamble", opts.CustomPrompt)
	} else {
		set("preamble", defaultPromptPreamble)
		var visible []string
		for _, name := range declaredTools {
			if snippet := opts.ToolSnippets[name]; snippet != "" {
				visible = append(visible, "- "+name+": "+snippet)
			}
		}
		tools := "(none)"
		if len(visible) > 0 {
			tools = strings.Join(visible, "\n")
		}
		set("tools", tools+"\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.")
		set("rules", buildRules(declaredTools, opts.ToolGuidelines, opts.PromptGuidelines))
		// Upstream always sets a docs section; the verified builder returns
		// "" when nothing exists, and an empty docs section is omitted rather
		// than rendered as an empty tag.
		if docs := docsSection(opts); docs != "" {
			set("docs", docs)
		}
	}

	if opts.AppendSystemPrompt != "" {
		set("addendum", opts.AppendSystemPrompt)
	}
	if len(opts.ContextFiles) > 0 {
		set("project_context", renderProjectContext(opts.ContextFiles))
	}
	// Skills are offered as soon as some tool can load their files, and the
	// block names that tool: the first of read and bash that is declared
	// (upstream 1d6dbf9e3). A hidden reader is still reachable through another
	// tool, so skills stay but the hint names no tool (upstream c30840c2e).
	skillFileReadTool := SkillFileReadTool("")
	switch {
	case slices.Contains(declaredTools, "read"):
		skillFileReadTool = SkillFileReadToolRead
	case slices.Contains(declaredTools, "bash"):
		skillFileReadTool = SkillFileReadToolBash
	case slices.Contains(opts.SelectedTools, "read") || slices.Contains(opts.SelectedTools, "bash"):
		skillFileReadTool = SkillFileReadToolIndirect
	}
	if skillFileReadTool != "" && len(opts.Skills) > 0 {
		if skills := jstext.Trim(FormatSkillsForPromptWithTool(opts.Skills, skillFileReadTool)); skills != "" {
			set("skills", skills)
		}
	}
	set("cwd", strings.ReplaceAll(opts.Cwd, `\`, "/"))
	for _, section := range customSections {
		if section.Value != nil && *section.Value != "" {
			set(section.Name, *section.Value)
		}
	}

	entries := promptSections.Entries()
	sections := make(ai.SystemSections, len(entries))
	for i, section := range entries {
		content := *section.Value
		if section.Name != "preamble" {
			content = "<" + section.Name + ">\n" + content + "\n</" + section.Name + ">"
		}
		sections[i] = ai.SystemSection{Name: section.Name, Value: &content}
	}
	return sections, nil
}

// docsTopic is one "When asked about" entry of the docs section: the topic and
// the files (or directories) it references, relative to the docs or examples
// root.
type docsTopic struct {
	topic string
	paths []string
}

// docsTopics is the default prompt's topic list, in upstream order, with the
// file references spelled exactly as the prompt renders them.
var docsTopics = []docsTopic{
	{"extensions", []string{"docs/extensions.md", "examples/extensions/"}},
	{"themes", []string{"docs/themes.md"}},
	{"skills", []string{"docs/skills.md"}},
	{"prompt templates", []string{"docs/prompt-templates.md"}},
	{"TUI components", []string{"docs/tui.md"}},
	{"keybindings", []string{"docs/keybindings.md"}},
	{"SDK integrations", []string{"docs/sdk.md"}},
	{"custom providers", []string{"docs/custom-provider.md"}},
	{"adding models", []string{"docs/models.md"}},
	{"pi packages", []string{"docs/packages.md"}},
	{"environment variables", []string{"docs/environment-variables.md"}},
	{"MCP servers", []string{"docs/mcp.md"}},
	{"codemode scripts and non-LLM models such as classifiers and image models", []string{"docs/codemode.md"}},
}

// docsSection is the default prompt's pi documentation section.
//
// Unverified (the upstream behavior) it renders the full documentation set
// byte for byte, paths interpolated. Verified (opts.VerifyDocPaths) it renders
// the same shape but advertises only what exists: the Main documentation line
// needs the README, each doc-directory line needs its directory, and each
// topic needs at least one of its referenced files or directories; an empty
// result omits the whole section.
func docsSection(opts BuildSystemPromptOptions) string {
	readmePath := opts.ReadmePath
	if readmePath == "" {
		readmePath = ReadmePath()
	}
	docsPath := opts.DocsPath
	if docsPath == "" {
		docsPath = DocsPath()
	}
	examplesPath := opts.ExamplesPath
	if examplesPath == "" {
		examplesPath = ExamplesPath()
	}
	if !opts.VerifyDocPaths {
		return `Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: ` + readmePath + `
- Additional docs: ` + docsPath + `
- Examples: ` + examplesPath + ` (extensions, custom tools, SDK)
- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md), MCP servers (docs/mcp.md), codemode scripts and non-LLM models such as classifiers and image models (docs/codemode.md)
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)`
	}

	resolveRef := func(ref string) string {
		if strings.HasPrefix(ref, "examples/") {
			return filepath.Join(examplesPath, ref[len("examples/"):])
		}
		return filepath.Join(docsPath, ref[len("docs/"):])
	}
	docsDir := isDir(docsPath)
	examplesDir := isDir(examplesPath)

	var lines []string
	if fileExists(readmePath) {
		lines = append(lines, "- Main documentation: "+readmePath)
	}
	if docsDir {
		lines = append(lines, "- Additional docs: "+docsPath)
	}
	if examplesDir {
		lines = append(lines, "- Examples: "+examplesPath+" (extensions, custom tools, SDK)")
	}
	if docsDir {
		lines = append(lines, "- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory")
	}
	topics := make([]string, 0, len(docsTopics))
	for _, topic := range docsTopics {
		for _, ref := range topic.paths {
			target := resolveRef(ref)
			if (strings.HasSuffix(ref, "/") && isDir(target)) || fileExists(target) {
				topics = append(topics, topic.topic+" ("+strings.Join(topic.paths, ", ")+")")
				break
			}
		}
	}
	if len(topics) > 0 {
		lines = append(lines, "- When asked about: "+strings.Join(topics, ", "))
	}
	if docsDir || examplesDir {
		lines = append(lines, "- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing")
	}
	if len(topics) > 0 || dirHasEntries(docsDir, docsPath) || dirHasEntries(examplesDir, examplesPath) {
		lines = append(lines, "- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)")
	}
	if len(lines) == 0 {
		return ""
	}
	return "Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):\n" + strings.Join(lines, "\n")
}

// isDir reports whether p exists and is a directory.
func isDir(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.IsDir()
}

// dirHasEntries reports whether the (existing) directory p holds at least one
// entry. exists false means false: the caller already knows the directory is
// absent.
func dirHasEntries(exists bool, p string) bool {
	if !exists {
		return false
	}
	entries, err := os.ReadDir(p)
	return err == nil && len(entries) > 0
}

// SystemPromptState is the complete prompt state a set of options describes
// (pi buildSystemPromptState's result): the content and sections of a system
// message holding the whole prompt. A forced prompt is opaque and lives in
// Content with nil Sections; otherwise Content is empty and Sections carry the
// structured prompt.
type SystemPromptState struct {
	Content  string
	Sections ai.SystemSections
}

// BuildSystemPromptState returns the complete prompt state for input (pi
// buildSystemPromptState, upstream e4c75a732): a non-nil ForceSystemPrompt is
// the whole content, verbatim; otherwise the sections are
// BuildSystemPromptSections', and so is the error.
func BuildSystemPromptState(input BuildSystemPromptOptions) (SystemPromptState, error) {
	if input.ForceSystemPrompt != nil {
		return SystemPromptState{Content: *input.ForceSystemPrompt}, nil
	}
	sections, err := BuildSystemPromptSections(input)
	if err != nil {
		return SystemPromptState{}, err
	}
	return SystemPromptState{Sections: sections}, nil
}

// BuildSystemPrompt renders the system prompt exactly as the transcript's
// system message replays it (pi buildSystemPrompt): BuildSystemPromptState's
// content followed by its sections, joined by a blank line. It fails when
// BuildSystemPromptState does.
func BuildSystemPrompt(opts BuildSystemPromptOptions) (string, error) {
	state, err := BuildSystemPromptState(opts)
	if err != nil {
		return "", err
	}
	message := ai.NewSystemText(state.Content, 0)
	message.Sections = state.Sections
	return ai.GetSystemMessageText(message), nil
}

// objectPrototypeKeys are the properties every plain JS object inherits from
// Object.prototype. Reading one of these names from an object that lacks it as
// an own property yields the inherited value, never undefined.
var objectPrototypeKeys = map[string]bool{
	"constructor": true, "__defineGetter__": true, "__defineSetter__": true, "hasOwnProperty": true,
	"__lookupGetter__": true, "__lookupSetter__": true, "isPrototypeOf": true, "propertyIsEnumerable": true,
	"toString": true, "valueOf": true, "__proto__": true, "toLocaleString": true,
}

// DiffSystemPromptSections diffs the sections the model currently has
// (replayed from the transcript) against the desired ones (pi
// diffSystemPromptSections). The patch sets every changed or new section and
// removes, with a nil value, every section current lacks; it reports false,
// with a nil patch, when nothing changed. The patch is in JS key order.
//
// pi compares plain JS objects, and Go follows their lookups: a previous
// section named like an Object.prototype property (e.g. "constructor") is never
// reported removed, because current's inherited property is not undefined, and
// a "__proto__" section is never patched, because assigning __proto__ sets the
// object's prototype instead of a key.
func DiffSystemPromptSections(previous, current ai.SystemSections) (ai.SystemSections, bool) {
	patch := ai.SystemSections{}
	for _, section := range current.Entries() {
		if section.Name == "__proto__" {
			continue
		}
		if value, ok := previous.Get(section.Name); !ok || !sameSectionValue(value, section.Value) {
			patch.Set(section.Name, section.Value)
		}
	}
	for _, section := range previous.Entries() {
		if _, ok := current.Get(section.Name); !ok && !objectPrototypeKeys[section.Name] {
			patch.Set(section.Name, nil)
		}
	}
	if patch.Len() == 0 {
		return nil, false
	}
	return patch, true
}

// sameSectionValue is JS strict equality over string | null.
func sameSectionValue(a, b *string) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}
