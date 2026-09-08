// Modified by Delta-AI under Apache 2.0
/**
 * Minimal Server-Sent Events parser over a fetch `ReadableStream` body.
 *
 * Handles the subset the gateway emits: `event:`/`data:` fields, multi-line
 * data, comment keep-alives (`: ...`), and CRLF/LF line endings.
 */

export interface RawSseEvent {
  /** SSE `event:` name (`undefined` = unnamed, the SSE default). */
  event: string | undefined;
  /** Concatenated `data:` lines. */
  data: string;
}

export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<RawSseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName: string | undefined;
  let dataLines: string[] = [];

  const dispatch = (): RawSseEvent | undefined => {
    if (dataLines.length === 0) {
      eventName = undefined;
      return undefined;
    }
    const event: RawSseEvent = { event: eventName, data: dataLines.join("\n") };
    eventName = undefined;
    dataLines = [];
    return event;
  };

  const processLine = (rawLine: string): RawSseEvent | undefined => {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "") {
      return dispatch();
    }
    if (line.startsWith(":")) {
      return undefined; // comment / keep-alive
    }
    const colon = line.indexOf(":");
    let field: string;
    let value: string;
    if (colon === -1) {
      field = line;
      value = "";
    } else {
      field = line.slice(0, colon);
      value = line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
    }
    if (field === "event") {
      eventName = value;
    } else if (field === "data") {
      dataLines.push(value);
    }
    // `id` / `retry` fields are ignored: the gateway never sets them.
    return undefined;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const rawLine = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        const event = processLine(rawLine);
        if (event) yield event;
      }
    }
    buffer += decoder.decode();
    if (buffer.length > 0) {
      const event = processLine(buffer);
      if (event) yield event;
    }
    const trailing = dispatch();
    if (trailing) yield trailing;
  } finally {
    reader.releaseLock();
  }
}
