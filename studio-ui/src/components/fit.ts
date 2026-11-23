import { useEffect, useRef, useState } from "react";

/** Largest width x height box with the given aspect ratio that fits inside the observed element. */
export function useFit(width: number, height: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const { width: cw, height: ch } = e.contentRect;
      const scale = Math.min(cw / width, ch / height);
      setSize({ w: Math.floor(width * scale), h: Math.floor(height * scale) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [width, height]);
  return { ref, size };
}
