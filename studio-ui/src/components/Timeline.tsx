import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";

/**
 * Frame strip with a draggable playhead, optional trim handles, coverage (frames where the actor
 * is tracked) and markers (frames with user selections). Always left-to-right like a video timeline.
 */
export function Timeline({
  count, current, onSeek, thumb, coverage, markers = [], trim, onTrim, limit,
}: {
  count: number; current: number; onSeek: (t: number) => void; thumb: (t: number) => string;
  coverage?: Set<number> | null; markers?: number[];
  trim?: { start: number; end: number }; onTrim?: (start: number, end: number) => void;
  limit?: { start: number; end: number };
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [drag, setDrag] = useState<null | "head" | "start" | "end">(null);
  const [localTrim, setLocalTrim] = useState(trim);
  useEffect(() => setLocalTrim(trim), [trim?.start, trim?.end]);

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  const tiles = Math.max(4, Math.min(count, Math.round(width / 72)));
  const tileFrames = useMemo(
    () => Array.from({ length: tiles }, (_, i) => Math.min(count - 1, Math.round((i + 0.5) * (count / tiles)))),
    [tiles, count],
  );
  const pct = (t: number) => (count <= 1 ? 0 : (t / (count - 1)) * 100);
  const frameAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    let t = Math.round(((clientX - r.left) / r.width) * (count - 1));
    t = Math.max(0, Math.min(count - 1, t));
    if (limit) t = Math.max(limit.start, Math.min(limit.end - 1, t));
    return t;
  };

  const coverageRuns = useMemo(() => {
    if (!coverage) return [];
    const runs: [number, number][] = [];
    let s = -1;
    for (let t = 0; t <= count; t++) {
      const on = coverage.has(t);
      if (on && s < 0) s = t;
      if (!on && s >= 0) { runs.push([s, t - 1]); s = -1; }
    }
    return runs;
  }, [coverage, count]);

  const move = (clientX: number) => {
    const t = frameAt(clientX);
    if (drag === "head") onSeek(t);
    else if (drag === "start" && localTrim) setLocalTrim({ start: Math.min(t, localTrim.end - 5), end: localTrim.end });
    else if (drag === "end" && localTrim) setLocalTrim({ start: localTrim.start, end: Math.max(t + 1, localTrim.start + 5) });
  };

  return (
    <div dir="ltr" className="select-none">
      <div
        ref={ref}
        className="relative h-16 cursor-pointer overflow-hidden rounded-xl bg-black/40 ring-1 ring-white/10"
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture(e.pointerId);
          const handle = (e.target as HTMLElement).dataset.handle as "start" | "end" | undefined;
          setDrag(handle ?? "head");
          if (!handle) onSeek(frameAt(e.clientX));
        }}
        onPointerMove={(e) => drag && move(e.clientX)}
        onPointerUp={() => {
          if ((drag === "start" || drag === "end") && localTrim) onTrim?.(localTrim.start, localTrim.end);
          setDrag(null);
        }}
      >
        <div className="absolute inset-0 flex">
          {tileFrames.map((t, i) => (
            <img key={i} src={thumb(t)} draggable={false} loading="lazy"
              className="pointer-events-none h-full min-w-0 flex-1 object-cover opacity-80" />
          ))}
        </div>
        {localTrim && (
          <>
            <div className="pointer-events-none absolute inset-y-0 left-0 bg-black/70" style={{ width: `${pct(localTrim.start)}%` }} />
            <div className="pointer-events-none absolute inset-y-0 right-0 bg-black/70" style={{ width: `${100 - pct(localTrim.end - 1)}%` }} />
            <div className="pointer-events-none absolute inset-y-0 border-y-2 border-violet-400/80"
              style={{ left: `${pct(localTrim.start)}%`, right: `${100 - pct(localTrim.end - 1)}%` }} />
            {onTrim && (["start", "end"] as const).map((h) => (
              <div key={h} data-handle={h}
                className="absolute inset-y-0 z-10 w-3 -translate-x-1/2 cursor-ew-resize rounded-md bg-violet-400 shadow-lg shadow-violet-500/40 hover:bg-violet-300"
                style={{ left: `${pct(h === "start" ? localTrim.start : localTrim.end - 1)}%` }} />
            ))}
          </>
        )}
        <div className="pointer-events-none absolute inset-y-0 z-20 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]"
          style={{ left: `${pct(current)}%` }}>
          <div className="absolute -top-1 left-1/2 size-3 -translate-x-1/2 rotate-45 rounded-sm bg-white" />
        </div>
      </div>
      {(coverage || markers.length > 0) && (
        <div className="relative mt-1.5 h-2.5">
          <div className="absolute inset-x-0 top-1 h-1 rounded-full bg-white/[0.06]" />
          {coverageRuns.map(([a, b]) => (
            <div key={a} className="absolute top-1 h-1 rounded-full bg-cyan-400/80"
              style={{ left: `${pct(a)}%`, width: `${Math.max(0.4, pct(b) - pct(a))}%` }} />
          ))}
          {markers.map((m) => (
            <button key={m} onClick={() => onSeek(m)}
              className={clsx("absolute top-0 size-2.5 -translate-x-1/2 rounded-full ring-2 ring-ink-950",
                m === current ? "bg-white" : "bg-emerald-400")} style={{ left: `${pct(m)}%` }} />
          ))}
        </div>
      )}
    </div>
  );
}
