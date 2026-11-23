import { useEffect, useState } from "react";

/** Steps through frames [start, end) at the clip's frame rate while playing. */
export function usePlayer(start: number, end: number, fps: number, initial?: number) {
  const [frame, setFrame] = useState(initial ?? start);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setFrame((f) => (f + 1 >= end ? start : f + 1)), 1000 / Math.min(fps || 16, 24));
    return () => clearInterval(id);
  }, [playing, start, end, fps]);
  useEffect(() => {
    if (frame < start || frame >= end) setFrame(Math.max(start, Math.min(end - 1, frame)));
  }, [start, end]); // eslint-disable-line react-hooks/exhaustive-deps
  return { frame, setFrame, playing, setPlaying };
}
