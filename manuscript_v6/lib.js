// Shared document helpers for manuscript V6 (copied from V5; captions in Springer style, section rows in tables).
const fs = require("fs");
const d = require("docx");
const {
  Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, ExternalHyperlink
} = d;

const INK = "000000";
const GREY = "3F3F3F";
const RULE = "8C8C8C";
const HEADSHADE = "E6E6E6";
const LINK = "28598A";
const W = 9360;
const FONT = "Times New Roman";

// Inline markup: **bold**, *italic*. Kept deliberately small.
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ t: text.slice(last, m.index) });
    const s = m[0];
    if (s.startsWith("**")) out.push({ t: s.slice(2, -2), b: true });
    else out.push({ t: s.slice(1, -1), i: true });
    last = m.index + s.length;
  }
  if (last < text.length) out.push({ t: text.slice(last) });
  return out.map(r => new TextRun({
    text: r.t, font: FONT, size: base.size || 22, bold: r.b || base.bold,
    italics: r.i || base.italics, color: base.color || INK
  }));
}

function p(text, o = {}) {
  return new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED,
    spacing: { after: o.after === undefined ? 140 : o.after, line: 276 },
    indent: o.indent,
    keepNext: o.keepNext,
    children: runs(text, o)
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, keepNext: true,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, font: FONT, size: 26, bold: true, color: INK })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2, keepNext: true,
    spacing: { before: 260, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 23, bold: true, color: INK })]
  });
}

function cell(text, o = {}) {
  return new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.head ? { type: ShadingType.CLEAR, fill: HEADSHADE, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: o.align || AlignmentType.LEFT,
      keepNext: !!o.keepNext,
      spacing: { after: 0, line: 240 },
      children: runs(String(text), { size: o.size || 18, bold: o.head || o.bold })
    })]
  });
}

function table(header, rows, widths, opts = {}) {
  const bold = opts.boldRows || [];
  const leftCols = opts.leftCols || 1;
  const size = opts.size || 18;
  const al = i => (i < leftCols ? AlignmentType.LEFT : AlignmentType.RIGHT);
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 8, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 8, color: RULE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "CFCFCF" },
      insideVertical: { style: BorderStyle.NONE }
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((t, i) => cell(t, { w: widths[i], head: true, align: al(i), size,
                                                  keepNext: opts.keep !== false }))
      }),
      ...rows.map((r, ri) => r && r.section !== undefined
        ? new TableRow({
            cantSplit: true,
            children: [new TableCell({
              width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
              columnSpan: widths.length,
              margins: { top: 80, bottom: 40, left: 90, right: 90 },
              children: [new Paragraph({ keepNext: true, spacing: { after: 0, line: 240 },
                children: runs(String(r.section), { size, bold: true }) })]
            })]
          })
        : new TableRow({
        cantSplit: true,
        children: r.map((t, i) => cell(t, { w: widths[i], bold: bold.includes(ri), align: al(i), size,
                                            keepNext: opts.keep !== false && ri < rows.length - 1 }))
      }))
    ]
  });
}

function caption(kind, n, text) {
  return new Paragraph({
    keepNext: kind === "Table",
    spacing: { before: kind === "Table" ? 200 : 0, after: kind === "Table" ? 90 : 240, line: 260 },
    children: [
      new TextRun({ text: `${kind === "Figure" ? "Fig." : kind} ${n}  `, font: FONT, size: 19, bold: true, color: INK }),
      ...runs(text, { size: 19, color: GREY })
    ]
  });
}

function figure(file, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true,
    spacing: { before: 220, after: 80 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
                              transformation: { width: w, height: h } })]
  });
}

function rule(after = 260) {
  return new Paragraph({
    spacing: { after },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 8 } },
    children: [new TextRun({ text: "", font: FONT, size: 2 })]
  });
}

// Reference entry with hanging indent; a trailing URL becomes a working link.
function ref(text) {
  const m = text.match(/(https?:\/\/\S+)$/);
  const head = m ? text.slice(0, m.index) : text;
  const children = runs(head, { size: 19 });
  if (m) {
    children.push(new ExternalHyperlink({
      link: m[1],
      children: [new TextRun({ text: m[1], font: FONT, size: 19, color: LINK,
                               underline: { type: "single", color: LINK } })]
    }));
  }
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { after: 70, line: 250 },
    indent: { left: 360, hanging: 360 },
    children
  });
}

module.exports = { d, p, h1, h2, table, caption, figure, rule, ref, runs, INK, GREY, FONT, W };
