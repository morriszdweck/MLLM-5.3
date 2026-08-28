# MLLM-5.3 Design System

## 0. Research Log

- Embedded refs: shortlisted Notion, Claude, and Linear in the design index; picked `minimalist-skill.md` + `notion.md` because the product is a document editor and needs warm editorial restraint rather than a chat-first shell.
- Lazyweb: 3 desktop queries, 3 screens viewed (Grammarly writing assistant, Coda workspace, Zed edit prediction) → took the persistent top context bar, document-first central measure, compact inspector/status rail, and explicit loading/empty/error states.
- Imagen drafts: `/Users/morrisdweck/.codex/generated_images/01a0464f-2720-7000-a865-fda61fe6dbf9/exec-b2996a49-cb6b-4f20-9dfa-3b95ad393ab9.png`, `/Users/morrisdweck/.codex/generated_images/01a0464f-2720-7000-a865-fda61fe6dbf9/exec-04ff4c2f-3da2-49bd-a1db-052a536842ad.png` → picked the first draft as the reference-fidelity contract because its paper canvas, rust continuation, and inspector map directly to this editor.
- Skipped lanes: none.

## 1. Atmosphere & Identity

MLLM-5.3 feels like a quiet writing desk with a small instrument panel attached: warm paper, dark ink, measured whitespace, and a suggestion that arrives without taking the page away. The signature is a rust-colored ghost continuation aligned to the writer's caret, paired with a narrow model inspector that makes local inference visible without turning the editor into a dashboard.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|------|------|------|------|
| Surface/canvas | `--surface-canvas` | `#F3EFE7` | `#151D1F` | App surround and breathing room |
| Surface/paper | `--surface-paper` | `#FFFDF8` | `#1D2628` | Writing surface |
| Surface/panel | `--surface-panel` | `#ECE7DE` | `#253033` | Inspector and status regions |
| Surface/ink | `--surface-ink` | `#1D2628` | `#0F1517` | Header and deep controls |
| Text/primary | `--text-primary` | `#24201C` | `#F7F1E8` | Prose and headings |
| Text/secondary | `--text-secondary` | `#6B6258` | `#C6BBAE` | Labels and explanations |
| Text/tertiary | `--text-tertiary` | `#9A9084` | `#968B80` | Hints, disabled, metadata |
| Border/default | `--border-default` | `#D8CEC1` | `#465154` | Dividers and controls |
| Border/subtle | `--border-subtle` | `#E6DED4` | `#354044` | Hairlines and quiet grouping |
| Accent/primary | `--accent-primary` | `#B85435` | `#D2704D` | Ghost text, focus, active controls |
| Accent/hover | `--accent-hover` | `#8F3F28` | `#F08C65` | Hover and pressed accent state |
| Status/success | `--status-success` | `#2F6E52` | `#8CC4A2` | Ready/accepted state |
| Status/warning | `--status-warning` | `#9A6A27` | `#E0B66B` | Slow or partial state |
| Status/error | `--status-error` | `#A8433E` | `#EF9B93` | Load and inference errors |
| Status/info | `--status-info` | `#4C7080` | `#9BC4D1` | Neutral model information |

### Rules

- `--accent-primary` is reserved for active inference and controls; it is never a decorative fill.
- Paper and panel are separated by tone plus one-pixel hairlines; no heavy shadows.
- Dark mode tokens are reserved for the header and system preference fallback. The release demo ships light first.
- No color is introduced in CSS without adding it to this table first.

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Tracking | Usage |
|------|------|--------|-------------|----------|------|
| Display | `48px / 3rem` | 700 | 1.1 | `-0.02em` | Brand title |
| H1 | `36px / 2.25rem` | 700 | 1.2 | `-0.015em` | Editor document title |
| H2 | `28px / 1.75rem` | 600 | 1.3 | `-0.01em` | Inspector section |
| H3 | `22px / 1.375rem` | 600 | 1.4 | 0 | Small panel heading |
| Body/lg | `18px / 1.125rem` | 400 | 1.6 | 0 | Writing canvas |
| Body | `16px / 1rem` | 400 | 1.6 | 0 | Default UI and document text |
| Body/sm | `14px / 0.875rem` | 400 | 1.5 | 0 | Secondary copy |
| Caption | `12px / 0.75rem` | 500 | 1.4 | `0.02em` | Model telemetry |
| Overline | `11px / 0.6875rem` | 600 | 1.3 | `0.08em` | Section labels |

### Font Stack

- Primary: `-apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif`.
- Serif: `Iowan Old Style, Baskerville, "Times New Roman", serif` for the writing surface and brand title.
- Mono: `SFMono-Regular, Menlo, Monaco, monospace` for telemetry and keyboard labels. The third family is intentional: it distinguishes machine state from human prose.

### Rules

- Body text never falls below 14px.
- The writing measure is capped at 68ch; headings use `clamp()` rather than forcing four-line wraps.
- Human text is serif in the editor; controls stay in the primary sans; model state stays mono.

## 4. Spacing & Layout

### Base Unit

All spacing derives from a base of 4px.

| Token | Value | Usage |
|------|------|------|
| `--space-1` | 4px | Icon-to-label and hairline gaps |
| `--space-2` | 8px | Compact controls and metadata |
| `--space-3` | 12px | Field padding and clusters |
| `--space-4` | 16px | Standard panel padding |
| `--space-5` | 20px | Comfortable control groups |
| `--space-6` | 24px | Canvas and inspector inner padding |
| `--space-8` | 32px | Major component separation |
| `--space-10` | 40px | Editorial breathing room |
| `--space-12` | 48px | Desktop canvas top rhythm |
| `--space-16` | 64px | Brand/header separation |

### Grid

- Max content width: 1440px.
- Desktop columns: fixed 236px outline rail, fluid document canvas, 312px inspector.
- Tablet: outline rail hides to preserve writing width; inspector stays visible as a 280px right rail.
- Mobile: one column; header and editor stay bounded to `100dvb`; inspector becomes a disclosure below the editor.

### Scroll Ownership

- `.app-shell` owns viewport height and never scrolls the page on desktop and tablet.
- `.editor-scroll` is the sole vertical scroll owner for document content on desktop and tablet; mobile lets the stacked workspace flow with the page.
- `.outline-rail` and `.inspector-scroll` each own their own overflow only when their content exceeds their bounded column; they never move the document scroll.
- Header, outline title row, inspector heading, and editor status bar remain fixed within their regions.

## 5. Components

### Application Shell

- **Structure**: `header.context-bar` → `main.workspace-grid` → `aside.outline-rail` + `section.editor-scroll` + `aside.inspector-scroll`.
- **Variants**: desktop split, tablet compressed, mobile stacked.
- **Spacing**: `--space-2` header, `--space-4` panel, `--space-12` canvas start.
- **States**: default, model-loading, ready, warning, error.
- **Accessibility**: landmarks, skip link, visible focus, no keyboard trap.
- **Motion**: inspector disclosure changes opacity and transform only; reduced motion removes transform.
- **Layout**: `scroll-body-shell` plus `fixed-sidenav-shell`; editor is the primary scroll owner.

### Editor Canvas

- **Structure**: document title, line-number gutter, `textarea` input, ghost continuation layer, suggestion tray, status bar.
- **Variants**: empty, writing, loading, completion-ready, completion-error.
- **Spacing**: `--space-6` inner padding, `--space-10` paragraph rhythm, 68ch measure.
- **States**: default, focus, typing, loading, ghost-ready, accepted, dismissed, error.
- **Accessibility**: labelled textarea, live status region, keyboard commands, text remains selectable and editable.
- **Motion**: ghost text fades in over the standard duration; suggestion tray uses `action-swap`-style blur/opacity; reduced motion snaps to visible state.
- **Layout**: `content-limiter` inside `.editor-scroll`.

### Model Inspector

- **Structure**: model select, parameter card, sampling range controls, recent context preview, state message.
- **Variants**: wide rail and mobile disclosure.
- **Spacing**: `--space-4` sections, `--space-2` control clusters.
- **States**: ready, loading, disabled, error, empty-context.
- **Accessibility**: every control has a label, values are exposed as text, and the mobile disclosure is a labelled button with `aria-expanded` and `aria-controls`.
- **Motion**: width stays static; mobile disclosure uses opacity/transform only and closes on Escape.
- **Layout**: `sidebar`; inspector scrolls independently on desktop.

### Status & Shortcut Cluster

- **Structure**: status dot, status copy, `kbd` keycaps, action button.
- **Variants**: ready, busy, accepted, dismissed, error.
- **Spacing**: `--space-1` key internals, `--space-2` clusters.
- **States**: default, hover, active, focus, disabled, loading, success, error.
- **Accessibility**: action buttons use semantic labels; live status is polite and non-repeating.
- **Motion**: action label swaps with a short opacity/blur crossfade; loader is a calm opacity pulse under reduced motion.
- **Layout**: `cluster` with wrapping before overflow.

### Primitive Showcase

- **Structure**: one route state selected with `?showcase=1`, showing buttons, fields, range control, keycaps, status, empty, loading, and error examples.
- **Variants**: default, hover, active, focus, disabled, loading, empty, error.
- **Spacing**: `--space-4` grid gaps and `--space-6` section padding.
- **States**: every listed state is rendered as an inspectable DOM primitive.
- **Accessibility**: showcase controls are keyboard reachable and visually labelled.
- **Motion**: the same production primitives are used; no mock-only styles.
- **Layout**: intrinsic grid using `minmax(min(16rem, 100%), 1fr)` with no horizontal overflow.

## 6. Motion & Interaction

### Timing

| Type | Duration | Easing | Usage |
|------|----------|--------|------|
| Micro | 120ms | ease-out | Press, focus accent, keycap feedback |
| Standard | 240ms | ease-in-out | Ghost arrival, inspector disclosure |
| Emphasis | 480ms | `cubic-bezier(0.16, 1, 0.3, 1)` | First-ready surface and model swap |

### Rules

- `action-swap` mechanism: accept/dismiss/status copy crossfades in place, avoiding layout jitter.
- `drawer` mechanism: mobile inspector disclosure is an in-flow panel toggled by the labelled menu button; it does not add an overlay or lock document scroll.
- `loader` mechanism: model loading uses a small wordmark pulse; reduced motion uses opacity only.
- Only `transform`, `opacity`, and filter are animated. No decorative motion occurs without a state change.
- `prefers-reduced-motion: reduce` disables transforms and timing loops.

## 7. Depth & Surface

### Strategy

Mixed: tonal shift is the primary hierarchy; whisper borders define interaction boundaries; shadows are limited to the suggestion tray.

| Level | Treatment | Usage |
|------|-----------|------|
| Flat | `--surface-canvas` | App surround |
| Paper | `--surface-paper` | Document surface |
| Panel | `--surface-panel` plus `--border-default` | Inspector and rails |
| Suggestion | `--surface-paper`, one soft `0 8px 20px` shadow using `--text-primary` at 8% | Floating ghost tray |

### Rules

- Borders are 1px and warm; no card has a heavy outline.
- The suggestion tray is the one elevated surface because it represents an actionable continuation.
- Rounded corners are limited to 4px controls and 8px suggestion tray.

## 8. Accessibility Constraints & Accepted Debt

### Constraints

- WCAG 2.2 AA target: 4.5:1 body contrast and 3:1 large-text contrast.
- The complete writing task is keyboard reachable: focus editor, wait for suggestion, Tab accept, Escape dismiss, Cmd/Ctrl+Enter accept suggestion.
- Every range, select, button, and disclosure has a visible label and focus indicator.
- No horizontal scroll at 375px; prose wraps naturally and long tokens use `overflow-wrap: anywhere`.
- Live status uses `aria-live="polite"`; model errors are persistent text, not color alone.
- Reduced-motion preferences are respected for every transition.

### Accepted Debt

| Item | Location | Why accepted | Owner / Exit |
|------|----------|--------------|--------------|
| Model files are large for a static demo | `site/models/` | The demo intentionally ships the trained checkpoints without a server or account layer | Release follow-up: add optional streamed shard delivery |
