import { Languages, Shirt } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type SystemInfo } from "../api";
import { useLang, useT } from "../i18n";

export function Logo({ small }: { small?: boolean }) {
  const t = useT();
  return (
    <div className="flex items-center gap-3">
      <div className={`brand-gradient flex items-center justify-center rounded-xl shadow-lg shadow-fuchsia-500/30 ${small ? "size-9" : "size-11"}`}>
        <Shirt className={small ? "size-5 text-white" : "size-6 text-white"} />
      </div>
      {!small && <span className="text-lg font-extrabold tracking-tight text-white">{t("appName")}</span>}
    </div>
  );
}

export function LangButton() {
  const t = useT();
  const toggle = useLang((s) => s.toggle);
  return (
    <button onClick={toggle} className="flex items-center gap-1.5 rounded-xl px-3 py-2 text-sm text-zinc-300 hover:bg-white/[0.06] hover:text-white">
      <Languages className="size-4" /> {t("language")}
    </button>
  );
}

export function GpuChip() {
  const t = useT();
  const [info, setInfo] = useState<SystemInfo | null>(null);
  useEffect(() => {
    let stop = false;
    const tick = () => api.system().then((s) => !stop && setInfo(s)).catch(() => {});
    tick();
    const id = setInterval(tick, 3000);
    return () => { stop = true; clearInterval(id); };
  }, []);
  if (!info?.gpu) return null;
  const busy = !!info.busy;
  return (
    <div className="hidden items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-zinc-300 md:flex" dir="ltr">
      <span className={`size-2 rounded-full ${busy ? "animate-pulse bg-amber-400" : "bg-emerald-400"}`} />
      <span>{info.gpu.name.replace("NVIDIA GeForce ", "")}</span>
      <span className="text-zinc-500">·</span>
      <span className="text-zinc-400">{busy ? t("gpuBusy") : t("gpuIdle")}</span>
    </div>
  );
}
