"use client";

import { ThemeProvider } from "next-themes";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import { load, save } from "@/lib/storage";
import type { AppConfig } from "@/lib/types";

/** Retrieval and display settings, shared by Chat and Self Study. */
export interface Settings {
  model: string;
  topK: number;
  rerankTopK: number;
  verify: boolean;
  debug: boolean;
}

interface AppContextValue {
  config: AppConfig | null;
  configError: string | null;
  settings: Settings;
  updateSettings: (patch: Partial<Settings>) => void;
}

const SETTINGS_KEY = "ep_settings";
const FALLBACK: Settings = { model: "", topK: 12, rerankTopK: 8, verify: true, debug: false };

const AppContext = createContext<AppContextValue | null>(null);

export function useApp(): AppContextValue {
  const value = useContext(AppContext);
  if (!value) throw new Error("useApp must be used inside <Providers>");
  return value;
}

function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings>(FALLBACK);

  useEffect(() => {
    api
      .get<AppConfig>("/api/config")
      .then((cfg) => {
        setConfig(cfg);
        let stored: Partial<Settings> = {};
        try {
          stored = JSON.parse(load(SETTINGS_KEY) ?? "{}");
        } catch {
          stored = {};
        }
        setSettings({
          ...FALLBACK,
          topK: cfg.defaults.top_k,
          rerankTopK: cfg.defaults.rerank_top_k,
          ...stored,
          // A model removed from the server since it was saved falls back.
          model:
            stored.model && cfg.available_models.includes(stored.model)
              ? stored.model
              : cfg.default_model,
        });
      })
      .catch((err: Error) => setConfigError(err.message));
  }, []);

  const updateSettings = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      save(SETTINGS_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ config, configError, settings, updateSettings }),
    [config, configError, settings, updateSettings],
  );
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <TooltipProvider delayDuration={300}>
        <AppStateProvider>{children}</AppStateProvider>
        <Toaster richColors position="bottom-right" />
      </TooltipProvider>
    </ThemeProvider>
  );
}
