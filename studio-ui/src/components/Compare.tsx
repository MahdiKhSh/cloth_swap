import { Pause, Play } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useFit } from "./fit";

function Divider({ pos, labels }: { pos: number; labels: [string, string] }) {
  return (
    <>
      <div className="pointer-events-none absolute inset-y-0 z-10 w-0.5 bg-white/90 shadow-[0_0_12px_rgba(0,0,0,0.6)]" style={{ left: `${pos}%` }}>
        <div className="absolute top-1/2 left-1/2 flex size-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-ink-900 shadow-xl">
          <span className="text-xs font-bold">⟷</span>
        </div>
      </div>
      <span className="pointer-events-none absolute top-3 left-3 rounded-full bg-black/60 px-2.5 py-1 text-[11px] font-semibold text-white">{labels[0]}</span>
      <span className="pointer-events-none absolute top-3 right-3 rounded-full bg-black/60 px-2.5 py-1 text-[11px] font-semibold text-white">{labels[1]}</span>
    </>
  );
}

function useSlider() {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState(false);
  const set = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
  };
  const handlers = {
    onPointerDown: (e: React.PointerEvent) => { (e.target as Element).setPointerCapture(e.pointerId); setDrag(true); set(e.clientX); },
    onPointerMove: (e: React.PointerEvent) => drag && set(e.clientX),
    onPointerUp: () => setDrag(false),
  };
  return { pos, ref, handlers };
}

/** Before/after image comparison: the "after" image is revealed on the right of the divider. */
export function CompareImage({ before, after, width, height, labels }: {
  before: string; after: string; width: number; height: number; labels: [string, string];
}) {
  const { pos, ref, handlers } = useSlider();
  const fit = useFit(width, height);
  return (
    <div ref={fit.ref} className="flex h-full min-h-0 w-full items-center justify-center">
      <div dir="ltr" ref={ref} {...handlers}
        className="relative shrink-0 cursor-ew-resize overflow-hidden rounded-2xl ring-1 ring-white/10 select-none"
        style={{ width: fit.size.w, height: fit.size.h }}>
        <img src={before} draggable={false} className="absolute inset-0 h-full w-full" />
        <img src={after} draggable={false} className="absolute inset-0 h-full w-full"
          style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
        <Divider pos={pos} labels={labels} />
      </div>
    </div>
  );
}

/** Two synchronised videos with a before/after divider and shared transport controls. */
export function CompareVideo({ before, after, offset, width, height, labels, extra }: {
  before: string; after: string; offset: number; width: number; height: number;
  labels: [string, string]; extra?: ReactNode;
}) {
  const { pos, ref, handlers } = useSlider();
  const fit = useFit(width, height);
  const a = useRef<HTMLVideoElement>(null);
  const b = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [dur, setDur] = useState(0);

  useEffect(() => {
    const master = b.current!;
    let raf = 0;
    const sync = () => {
      const slave = a.current;
      if (slave && Math.abs(slave.currentTime - (master.currentTime + offset)) > 0.08) slave.currentTime = master.currentTime + offset;
      setTime(master.currentTime);
      raf = requestAnimationFrame(sync);
    };
    raf = requestAnimationFrame(sync);
    return () => cancelAnimationFrame(raf);
  }, [offset, after]);

  const toggle = () => {
    const m = b.current!, s = a.current!;
    if (m.paused) { m.play(); s.play(); setPlaying(true); } else { m.pause(); s.pause(); setPlaying(false); }
  };

  return (
    <div className="flex h-full flex-col gap-3">
      <div ref={fit.ref} className="flex min-h-0 flex-1 items-center justify-center">
        <div dir="ltr" ref={ref} {...handlers}
          className="relative shrink-0 cursor-ew-resize overflow-hidden rounded-2xl bg-black ring-1 ring-white/10 select-none"
          style={{ width: fit.size.w, height: fit.size.h }}>
          <video ref={a} src={before} muted playsInline preload="auto" className="absolute inset-0 h-full w-full" />
          <video ref={b} src={after} playsInline preload="auto" loop className="absolute inset-0 h-full w-full"
            style={{ clipPath: `inset(0 0 0 ${pos}%)` }}
            onLoadedMetadata={(e) => setDur(e.currentTarget.duration)}
            onEnded={() => setPlaying(false)} />
          <Divider pos={pos} labels={labels} />
        </div>
      </div>
      <div dir="ltr" className="glass flex items-center gap-3 rounded-2xl px-3 py-2">
        <button onClick={toggle} className="flex size-9 items-center justify-center rounded-full brand-gradient text-white shadow-lg">
          {playing ? <Pause className="size-4" /> : <Play className="size-4 translate-x-px" />}
        </button>
        <input type="range" className="range flex-1" min={0} max={dur || 1} step={0.01} value={time}
          onChange={(e) => { const v = Number(e.target.value); b.current!.currentTime = v; a.current!.currentTime = v + offset; }} />
        <span className="w-24 text-end text-xs tabular-nums text-zinc-400">{time.toFixed(1)} / {dur.toFixed(1)}s</span>
        {extra}
      </div>
    </div>
  );
}
