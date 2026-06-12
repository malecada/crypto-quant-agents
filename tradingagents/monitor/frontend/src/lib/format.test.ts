import { describe, expect, it } from "vitest";
import { fmtUsd, fmtPct, fmtNum } from "./format";

describe("format", () => {
  it("fmtUsd", () => {
    expect(fmtUsd(10234.567)).toBe("$10,234.57");
    expect(fmtUsd(null)).toBe("—");
  });
  it("fmtPct from fraction", () => {
    expect(fmtPct(-0.0497)).toBe("-4.97%");
    expect(fmtPct(null)).toBe("—");
  });
  it("fmtNum", () => {
    expect(fmtNum(3.178, 2)).toBe("3.18");
    expect(fmtNum(null)).toBe("—");
  });
});
