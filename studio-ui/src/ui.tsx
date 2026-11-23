import clsx from "clsx";
import { Loader2, X } from "lucide-react";
import { useEffect, type ButtonHTMLAttributes, type ReactNode } from "react";
import { create } from "zustand";

type Variant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "secondary",
  size = "md",
  loading,
  icon,
  className,
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: ReactNode;
}) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all select-none",
        "disabled:opacity-45 disabled:cursor-not-allowed active:scale-[0.98]",
        size === "sm" && "h-8 px-3 text-xs",
        size === "md" && "h-10 px-4 text-sm",
        size === "lg" && "h-12 px-6 text-base",
        variant === "primary" &&
          "brand-gradient text-white shadow-lg shadow-fuchsia-500/20 hover:shadow-fuchsia-500/40 hover:brightness-110",
        variant === "secondary" && "bg-white/[0.06] text-zinc-100 border border-white/10 hover:bg-white/10",
        variant === "ghost" && "text-zinc-300 hover:bg-white/[0.06] hover:text-white",
        variant === "danger" && "bg-rose-500/10 text-rose-300 border border-rose-500/25 hover:bg-rose-500/20",
        className,
      )}
    >
      {loading ? <Loader2 className="size-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx("glass rounded-2xl", className)}>{children}</div>;
}

export function SectionTitle({ title, hint, right }: { title: string; hint?: string; right?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 mb-5 animate-fade-in">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-white">{title}</h1>
        {hint && <p className="mt-1.5 max-w-3xl text-sm leading-7 text-zinc-400">{hint}</p>}
      </div>
      {right}
    </div>
  );
}

export function Badge({ tone = "neutral", children }: { tone?: "neutral" | "brand" | "ok" | "warn"; children: ReactNode }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold",
        tone === "neutral" && "bg-white/[0.07] text-zinc-300",
        tone === "brand" && "bg-violet-500/15 text-violet-200",
        tone === "ok" && "bg-emerald-500/15 text-emerald-300",
        tone === "warn" && "bg-amber-500/15 text-amber-300",
      )}
    >
      {children}
    </span>
  );
}

export function ProgressBar({ value, active }: { value: number; active?: boolean }) {
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-white/[0.07]">
      <div
        className="absolute inset-y-0 start-0 rounded-full brand-gradient transition-[width] duration-500"
        style={{ width: `${Math.max(2, value * 100)}%` }}
      />
      {active && <div className="absolute inset-0 progress-shimmer animate-shimmer" />}
    </div>
  );
}

export function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-4 rounded-xl p-3 hover:bg-white/[0.03]">
      <span>
        <span className="block text-sm font-medium text-zinc-100">{label}</span>
        {hint && <span className="mt-0.5 block text-xs text-zinc-500">{hint}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={clsx("relative h-6 w-11 shrink-0 rounded-full transition-colors", checked ? "brand-gradient" : "bg-white/15")}
      >
        <span
          className={clsx(
            "absolute top-0.5 size-5 rounded-full bg-white shadow transition-all",
            checked ? "start-[22px]" : "start-0.5",
          )}
        />
      </button>
    </label>
  );
}

export function Slider({ label, value, min, max, step, onChange, format }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <div className="p-3">
      <div className="mb-2 flex items-center justify-between text-sm">
        <span className="text-zinc-200">{label}</span>
        <span className="tabular-nums text-zinc-400">{format ? format(value) : value}</span>
      </div>
      <input type="range" className="range w-full" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-zinc-200">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-zinc-500">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-xl border border-white/10 bg-black/30 px-3.5 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition focus:border-violet-400/60 focus:ring-4 focus:ring-violet-500/15";

export function Segmented<T extends string | number>({ value, options, onChange }: {
  value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-xl border border-white/10 bg-black/30 p-1">
      {options.map((o) => (
        <button key={String(o.value)} type="button" onClick={() => onChange(o.value)}
          className={clsx("rounded-lg px-3 py-1.5 text-sm transition",
            o.value === value ? "bg-white/12 text-white shadow" : "text-zinc-400 hover:text-zinc-200")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Modal({ open, onClose, children, wide }: { open: boolean; onClose: () => void; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm animate-fade-in" onClick={onClose}>
      <div className={clsx("glass relative max-h-full w-full overflow-auto rounded-3xl bg-ink-900/90 p-5 animate-pop", wide ? "max-w-6xl" : "max-w-lg")}
        onClick={(e) => e.stopPropagation()}>
        <button className="absolute end-4 top-4 rounded-lg p-1.5 text-zinc-400 hover:bg-white/10 hover:text-white" onClick={onClose}>
          <X className="size-5" />
        </button>
        {children}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ toasts
interface Toast { id: number; text: string; tone: "error" | "ok" | "info" }
export const useToasts = create<{ toasts: Toast[]; push: (text: string, tone?: Toast["tone"]) => void; drop: (id: number) => void }>(
  (set, get) => ({
    toasts: [],
    push: (text, tone = "info") => {
      const id = Date.now() + Math.random();
      set({ toasts: [...get().toasts, { id, text, tone }] });
      setTimeout(() => get().drop(id), 5000);
    },
    drop: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
  }),
);

export function Toasts() {
  const { toasts, drop } = useToasts();
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-5 z-[60] flex flex-col items-center gap-2">
      {toasts.map((t) => (
        <div key={t.id} onClick={() => drop(t.id)}
          className={clsx("pointer-events-auto glass cursor-pointer rounded-xl px-4 py-2.5 text-sm shadow-2xl animate-pop",
            t.tone === "error" && "border-rose-500/40 text-rose-200",
            t.tone === "ok" && "border-emerald-500/40 text-emerald-200")}>
          {t.text}
        </div>
      ))}
    </div>
  );
}

export function EmptyState({ icon, text }: { icon: ReactNode; text: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-white/10 p-10 text-center text-sm text-zinc-500">
      <div className="text-zinc-600">{icon}</div>
      {text}
    </div>
  );
}
