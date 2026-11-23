import { AlertTriangle, Pause, Play } from "lucide-react";
import { api, urls } from "../api";
import { FrameViewer } from "../components/FrameViewer";
import { Timeline } from "../components/Timeline";
import { useKey } from "../hooks";
import { useNum, useT } from "../i18n";
import { StepFooter, useWs } from "../pages/Workspace";
import { Button, Card, SectionTitle, useToasts } from "../ui";
import { usePlayer } from "./usePlayer";

export function StepVideo() {
  const t = useT();
  const num = useNum();
  const push = useToasts((s) => s.push);
  const { project: p, setProject } = useWs();
  const v = p.video!;
  const n = v.num_frames!;
  const trim = p.trim ?? { start: 0, end: n };
  const player = usePlayer(trim.start, trim.end, v.fps!);
  useKey((e) => {
    if (e.key === " ") { e.preventDefault(); player.setPlaying(!player.playing); }
  }, [player.playing]);

  const info = [
    { label: t("resolution"), value: `${v.width}×${v.height}` },
    { label: t("fps"), value: num(Math.round(v.fps! * 100) / 100) },
    { label: t("duration"), value: `${num(Math.round((n / v.fps!) * 10) / 10)}s` },
    { label: t("selectedFrames"), value: `${num(trim.end - trim.start)} / ${num(n)}` },
  ];

  const setTrim = (start: number, end: number) =>
    api.patch(p.id, { trim: { start, end } }).then(setProject).catch((e) => push(e.message, "error"));

  return (
    <div className="flex h-full flex-col gap-4">
      <SectionTitle title={t("videoTitle")} hint={t("videoHint")} />
      <div className="flex min-h-0 flex-1 gap-5">
        <FrameViewer className="min-h-0 flex-1" src={urls.frame(p.id, player.frame)} width={v.width!} height={v.height!} />
        <div className="w-64 shrink-0 space-y-3">
          {info.map((i) => (
            <Card key={i.label} className="p-4 animate-fade-in">
              <div className="text-xs text-zinc-500">{i.label}</div>
              <div className="mt-1 text-lg font-bold tabular-nums text-white" dir="ltr">{i.value}</div>
            </Card>
          ))}
          {p.actor.tracked && (
            <div className="flex gap-2 rounded-xl border border-amber-500/20 bg-amber-500/[0.06] p-3 text-xs leading-5 text-amber-200">
              <AlertTriangle className="size-4 shrink-0" /> {t("trimChangedWarning")}
            </div>
          )}
        </div>
      </div>
      <div className="flex items-center gap-3">
        <Button variant="secondary" className="size-10 !px-0" onClick={() => player.setPlaying(!player.playing)}>
          {player.playing ? <Pause className="size-4" /> : <Play className="size-4" />}
        </Button>
        <div className="flex-1">
          <Timeline count={n} current={player.frame} onSeek={(f) => { player.setPlaying(false); player.setFrame(f); }}
            thumb={(f) => urls.thumb(p.id, f)} trim={trim} onTrim={setTrim} />
        </div>
        <span className="w-28 text-end text-xs tabular-nums text-zinc-400" dir="ltr">
          {t("frame")} {player.frame} · {(player.frame / v.fps!).toFixed(2)}s
        </span>
      </div>
      <StepFooter next="actor" />
    </div>
  );
}
