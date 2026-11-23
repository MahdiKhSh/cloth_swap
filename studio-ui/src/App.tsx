import { useEffect, useState } from "react";
import { applyDir, useLang } from "./i18n";
import { Home } from "./pages/Home";
import { Workspace, type StepId } from "./pages/Workspace";
import { Toasts } from "./ui";

/** "#/" = projects, "#/p/<id>/<step>" = workspace */
function parse(hash: string): { project?: string; step?: StepId } {
  const m = hash.match(/^#\/p\/([^/]+)(?:\/([a-z]+))?/);
  return m ? { project: m[1], step: m[2] as StepId | undefined } : {};
}

export function navigate(project?: string, step?: string) {
  window.location.hash = project ? `#/p/${project}${step ? `/${step}` : ""}` : "#/";
}

export function App() {
  const lang = useLang((s) => s.lang);
  const [route, setRoute] = useState(parse(window.location.hash));
  useEffect(() => applyDir(lang), [lang]);
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  return (
    <>
      {route.project ? <Workspace key={route.project} id={route.project} step={route.step} /> : <Home />}
      <Toasts />
    </>
  );
}
