export interface Sam3Field {
  key: string;
  default: number | string | boolean;
  minimum: number | null;
  maximum: number | null;
  group: string;
  effect: string;
  help: string;
  type: "boolean" | "select" | "integer" | "number";
  options: string[] | null;
}
export interface Sam3Controls {
  groups: Record<string, string>;
  fields: Sam3Field[];
  values: Record<string, number | string | boolean>;
  overrides: Record<string, number | string | boolean>;
}

export interface GarmentBible {
  id: string;
  revision: number;
  created: number;
  updated: number;
  name: string;
  description: string;
  type: string;
  color: string;
  fabric: string;
  pattern: string;
  fit: string;
  collar: string;
  sleeves: string;
  closures: string;
  pockets: string;
  logo: string;
  seams: string;
  notes: string;
  images: { view: "front" | "back" | "side" | "detail"; file: string }[];
}

export interface Prompt {
  frame: number;
  text?: string;
  points: number[][];
  labels: number[];
  box: number[] | null;
}

export interface MaskLayer {
  name: string;
  prompt: string;
  operation: "add" | "subtract" | "ignore";
  selection?: (Prompt & { candidate?: string | null }) | null;
}

export interface Candidate {
  id: string;
  frame: number;
  seed?: number;
  prompt?: string;
  source: "ai" | "upload";
  created: number;
  /** overlap of the edited person with the original silhouette; low = the editor re-posed the person */
  match?: number;
}

export interface Job {
  id: string;
  project_id: string;
  kind: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  message: string;
  error: string | null;
  result?: { candidates?: string[]; [key: string]: unknown } | null;
  created: number;
  started: number | null;
  finished: number | null;
  logs?: string[];
}

export interface Project {
  id: string;
  name: string;
  created: number;
  updated: number;
  video: { file: string; fps?: number; width?: number; height?: number; num_frames?: number } | null;
  trim: { start: number; end: number } | null;
  actor: { prompts: Prompt[]; tracked: boolean; parts_ready?: boolean; parts_quality?: { visible_frames: number; face_frames: number; hair_frames: number; skipped_frames: number; skipped_indices: number[]; edited_pixels: number; protected_pixels: number } | null; visible: number[]; trim: number[] | null };
  mask_layers: { rules: MaskLayer[]; ready: boolean; tracked?: boolean; base_actor?: boolean };
  garment: { images: string[]; description: string; bible_id?: string | null; bible_revision?: number | null };
  prompt: { subject: string; scene: string; full?: string };
  keyframe: {
    frame: number | null;
    suggestions: { frame: number; score: number }[];
    candidates: Candidate[];
    chosen: string | null;
    prompt: string;
    extra?: string;
  };
  generate: { preset: string; steps: number | null; seed: number; relight: boolean; upscale: string; output?: string };
  composite: { mode: string; feather: number; dilate: number };
  result: { file: string; updated: number; preset?: string; size?: number[]; stale?: boolean; stale_reason?: "source" | "composite" | null } | null;
  jobs?: Job[];
}

export interface SystemInfo {
  backend?: string;
  model_ready?: boolean;
  gpu: { name: string; memory_gb: number } | null;
  busy: Job | null;
  presets: Record<string, { target_area: number; steps: number }>;
  relight_available: boolean;
}

export type Polygons = number[][][];

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

const P = (id: string) => `/api/projects/${id}`;

export const api = {
  system: () => request<SystemInfo>("GET", "/api/system"),
  garmentBibles: () => request<GarmentBible[]>("GET", "/api/garment-bibles"),
  createGarmentBible: (values: Partial<GarmentBible> & { from_project_id?: string }) =>
    request<GarmentBible>("POST", "/api/garment-bibles", values),
  updateGarmentBible: (id: string, values: Partial<GarmentBible>, expectedRevision: number) =>
    request<GarmentBible>("PATCH", `/api/garment-bibles/${id}`, { ...values, _expected_revision: expectedRevision }),
  uploadBibleImage: (id: string, view: string, file: File, expectedRevision: number) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("view", view);
    fd.append("expected_revision", String(expectedRevision));
    return request<GarmentBible>("POST", `/api/garment-bibles/${id}/images`, fd);
  },
  applyGarmentBible: (pid: string, bid: string) =>
    request<Project>("POST", `${P(pid)}/garment-bible/${bid}`),
  syncGarmentBible: (bid: string) =>
    request<{ revision: number; synced: string[]; skipped_running: string[] }>("POST", `/api/garment-bibles/${bid}/sync`),
  projects: () => request<Project[]>("GET", "/api/projects"),
  project: (id: string) => request<Project>("GET", P(id)),
  sam3Settings: (id: string) => request<Sam3Controls>("GET", `${P(id)}/sam3-settings`),
  saveSam3Settings: (id: string, values: Record<string, number | string | boolean>) =>
    request<Project>("PATCH", `${P(id)}/sam3-settings`, values),
  resetSam3Settings: (id: string) => request<Project>("POST", `${P(id)}/sam3-settings/reset`),
  patch: (id: string, body: Partial<Record<string, unknown>>) => request<Project>("PATCH", P(id), body),
  remove: (id: string) => request<{ ok: boolean }>("DELETE", P(id)),
  detect: (id: string, frame: number, text?: string) =>
    request<{ people: { box: number[]; area: number; polygons: Polygons }[] }>("POST", `${P(id)}/detect`, { frame, text }),
  segment: (id: string, prompt: Prompt & { candidate?: string | null; isolated?: boolean }) => request<{ polygons: Polygons; area: number }>("POST", `${P(id)}/segment`, prompt),
  setPrompts: (id: string, prompts: Prompt[]) => request<Project>("PUT", `${P(id)}/actor/prompts`, { prompts }),
  track: (id: string) => request<Job>("POST", `${P(id)}/actor/track`),
  separateParts: (id: string) => request<Job>("POST", `${P(id)}/actor/person-parts`),
  actorMask: (id: string, t: number, part: string = "actor") => request<{ polygons: Polygons }>("GET", `${P(id)}/actor/mask/${t}?part=${encodeURIComponent(part)}`),
  setMaskLayers: (id: string, layers: MaskLayer[], baseActor: boolean = true) => request<Project>("PUT", `${P(id)}/mask-layers`, { layers, base_actor: baseActor }),
  buildMaskLayers: (id: string) => request<Job>("POST", `${P(id)}/mask-layers/build`),
  composeMaskLayers: (id: string) => request<Job>("POST", `${P(id)}/mask-layers/compose`),
  addGarment: (id: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<Project>("POST", `${P(id)}/garments`, fd);
  },
  removeGarment: (id: string, name: string) => request<Project>("DELETE", `${P(id)}/garments/${name}`),
  generateKeyframes: (id: string, body: { frame: number; count: number; extra?: string; seed?: number }) =>
    request<Job>("POST", `${P(id)}/keyframe/generate`, body),
  instruction: (id: string, extra: string) =>
    request<{ text: string }>("GET", `${P(id)}/keyframe/instruction?extra=${encodeURIComponent(extra)}`),
  uploadKeyframe: (id: string, frame: number, file: File) => {
    const fd = new FormData();
    fd.append("frame", String(frame));
    fd.append("file", file);
    return request<Job>("POST", `${P(id)}/keyframe/upload`, fd);
  },
  chooseKeyframe: (id: string, candidate: string) => request<Project>("POST", `${P(id)}/keyframe/choose`, { candidate }),
  deleteCandidate: (id: string, cid: string) => request<Project>("DELETE", `${P(id)}/keyframe/candidates/${cid}`),
  generate: (id: string) => request<Job>("POST", `${P(id)}/generate`),
  composite: (id: string) => request<Job>("POST", `${P(id)}/composite`),
  job: (jobId: string) => request<Job>("GET", `/api/jobs/${jobId}`),
  cancel: (jobId: string) => request<Job>("POST", `/api/jobs/${jobId}/cancel`),
};

export const urls = {
  frame: (id: string, t: number) => `${P(id)}/frames/${t}.jpg`,
  thumb: (id: string, t: number) => `${P(id)}/thumbs/${t}.jpg`,
  isolated: (id: string, t: number) => `${P(id)}/actor/isolated/${t}.jpg`,
  source: (id: string) => `${P(id)}/source`,
  result: (id: string, v: number) => `${P(id)}/result?v=${Math.round(v)}`,
  download: (id: string) => `${P(id)}/result/download`,
  garment: (id: string, name: string) => `${P(id)}/garments/${name}`,
  bibleImage: (id: string, name: string, revision: number) => `/api/garment-bibles/${id}/images/${name}?v=${revision}`,
  candidate: (id: string, cid: string) => `${P(id)}/keyframe/candidates/${cid}.jpg`,
};

/** Upload a new project with progress reporting (fetch has no upload progress). */
export function createProject(file: File, name: string, onProgress: (f: number) => void) {
  return new Promise<{ project: Project; job: Job }>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", name);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () =>
      xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new ApiError(xhr.status, xhr.responseText));
    xhr.onerror = () => reject(new ApiError(0, "network"));
    xhr.open("POST", "/api/projects");
    xhr.send(fd);
  });
}
