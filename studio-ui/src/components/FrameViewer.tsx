import clsx from "clsx";
import { useRef, useState, type ReactNode } from "react";
import type { Polygons } from "../api";
import { useFit } from "./fit";

export interface Layer {
  polygons: Polygons;
  className: string;
  onClick?: () => void;
  key?: string;
}

export type Tool = "add" | "remove" | "box" | null;

export function polygonsPath(polygons: Polygons) {
  return polygons.map((poly) => "M" + poly.map(([x, y]) => `${x},${y}`).join("L") + "Z").join(" ");
}

/** A video frame with SVG overlays in image coordinates, and click / box input. */
export function FrameViewer({
  src, width, height, layers = [], points = [], box = null, tool = null,
  onPoint, onBox, children, className,
}: {
  src: string; width: number; height: number; layers?: Layer[];
  points?: { x: number; y: number; label: number }[]; box?: number[] | null; tool?: Tool;
  onPoint?: (x: number, y: number, label: number) => void; onBox?: (box: number[]) => void;
  children?: ReactNode; className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const fit = useFit(width, height);
  const [drag, setDrag] = useState<number[] | null>(null);
  const [loaded, setLoaded] = useState<string | null>(null);

  const toImage = (e: React.PointerEvent | React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return [
      Math.round(Math.min(Math.max(((e.clientX - r.left) / r.width) * width, 0), width - 1)),
      Math.round(Math.min(Math.max(((e.clientY - r.top) / r.height) * height, 0), height - 1)),
    ];
  };
  const r = width / 110;
  const hasClickLayer = layers.some((l) => l.onClick);

  return (
    <div ref={fit.ref} className={clsx("relative flex min-h-0 min-w-0 items-center justify-center", className)} dir="ltr">
      <div
        ref={ref}
        className={clsx("relative shrink-0 overflow-hidden rounded-2xl bg-black shadow-2xl shadow-black/50 ring-1 ring-white/10",
          tool && !hasClickLayer && "cursor-crosshair")}
        style={{ width: fit.size.w, height: fit.size.h, touchAction: "none" }}
        onPointerCancel={() => setDrag(null)}
        onContextMenu={(e) => {
          e.preventDefault();
          if (tool && tool !== "box" && onPoint && !hasClickLayer) {
            const [x, y] = toImage(e);
            onPoint(x, y, 0);
          }
        }}
        onPointerDown={(e) => {
          if (e.button !== 0 || hasClickLayer || !tool) return;
          const [x, y] = toImage(e);
          if (tool === "box") {
            e.currentTarget.setPointerCapture(e.pointerId);
            setDrag([x, y, x, y]);
          } else onPoint?.(x, y, tool === "add" ? 1 : 0);
        }}
        onPointerMove={(e) => {
          if (!drag) return;
          const [x, y] = toImage(e);
          setDrag([drag[0], drag[1], x, y]);
        }}
        onPointerUp={() => {
          if (!drag) return;
          const [x0, y0, x1, y1] = drag;
          setDrag(null);
          if (Math.abs(x1 - x0) > 8 && Math.abs(y1 - y0) > 8)
            onBox?.([Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]);
        }}
      >
        <img src={src} draggable={false} onLoad={() => setLoaded(src)}
          className={clsx("block h-full w-full select-none transition-opacity duration-200", loaded === src ? "opacity-100" : "opacity-80")} />
        <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          {layers.map((l, i) => (
            <path key={l.key ?? i} d={polygonsPath(l.polygons)} className={l.className}
              onClick={(e) => { e.stopPropagation(); l.onClick?.(); }} />
          ))}
          {(drag || box) && (() => {
            const [x0, y0, x1, y1] = drag ?? box!;
            return <rect x={Math.min(x0, x1)} y={Math.min(y0, y1)} width={Math.abs(x1 - x0)} height={Math.abs(y1 - y0)}
              fill="rgba(167,139,250,0.08)" stroke="#c4b5fd" strokeWidth={2} strokeDasharray="8 6" vectorEffect="non-scaling-stroke" rx={r / 2} />;
          })()}
          {points.map((p, i) => (
            <g key={i} transform={`translate(${p.x},${p.y})`} className="pointer-events-none">
              <circle r={r * 1.35} fill="rgba(0,0,0,0.35)" />
              <circle r={r} fill={p.label ? "#22c55e" : "#f43f5e"} stroke="white" strokeWidth={2} vectorEffect="non-scaling-stroke" />
              <path d={p.label ? `M${-r / 2},0H${r / 2}M0,${-r / 2}V${r / 2}` : `M${-r / 2},0H${r / 2}`}
                stroke="white" strokeWidth={2.5} strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            </g>
          ))}
        </svg>
        {children}
      </div>
    </div>
  );
}
