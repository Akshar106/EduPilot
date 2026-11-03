"use client";

import { MessageSquarePlus, Trash2 } from "lucide-react";

import { useApp } from "@/components/providers";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { relativeTime } from "@/lib/format";
import type { SessionSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SidebarProps {
  sessions: SessionSummary[];
  activeId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export function ChatSidebar({ sessions, activeId, onSelect, onNew, onDelete }: SidebarProps) {
  return (
    <div className="flex h-full flex-col gap-3 p-3">
      <Button onClick={onNew} variant="outline" className="justify-start">
        <MessageSquarePlus /> New conversation
      </Button>

      <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto">
        <div className="px-2 pb-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          Conversations
        </div>
        {sessions.length === 0 && <p className="px-2 text-sm text-muted-foreground">No conversations yet.</p>}
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={cn(
              "group flex items-center gap-1 rounded-lg pr-1 hover:bg-muted",
              s.session_id === activeId && "bg-muted",
            )}
          >
            <button
              type="button"
              onClick={() => onSelect(s.session_id)}
              className="min-w-0 flex-1 px-2 py-1.5 text-left"
            >
              <div className="truncate text-sm">{s.title || `Chat ${s.session_id.slice(0, 8)}`}</div>
              <div className="text-xs text-muted-foreground">{relativeTime(s.updated_at)}</div>
            </button>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label="Delete conversation"
                  className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                >
                  <Trash2 />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
                  <AlertDialogDescription>
                    Its messages and memory are removed permanently.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction variant="destructive" onClick={() => onDelete(s.session_id)}>
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        ))}
      </div>

      <Separator />
      <SettingsPanel />
    </div>
  );
}

/** Model and retrieval settings; persisted per browser. */
export function SettingsPanel({ compact = false }: { compact?: boolean }) {
  const { config, settings, updateSettings } = useApp();
  if (!config) return null;
  const groq = new Set(config.groq_models);
  const groups = [
    { label: "Groq — free and fast", models: config.available_models.filter((m) => groq.has(m)) },
    { label: "Gemini", models: config.available_models.filter((m) => !groq.has(m)) },
  ].filter((g) => g.models.length);

  return (
    <div className="space-y-3">
      <div className="px-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">Settings</div>
      <div className="space-y-1.5">
        <Label htmlFor="model">Model</Label>
        <Select value={settings.model} onValueChange={(model) => updateSettings({ model })}>
          <SelectTrigger id="model" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {groups.map((g) => (
              <SelectGroup key={g.label}>
                <SelectLabel>{g.label}</SelectLabel>
                {g.models.map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectGroup>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="flex justify-between" title="Candidates pulled from the index before reranking">
          Retrieval top-K <span className="text-muted-foreground">{settings.topK}</span>
        </Label>
        <Slider min={4} max={20} step={1} value={[settings.topK]} onValueChange={([topK]) => updateSettings({ topK })} />
      </div>
      <div className="space-y-2">
        <Label className="flex justify-between" title="Excerpts handed to the model — the main lever on answer depth">
          Evidence chunks <span className="text-muted-foreground">{settings.rerankTopK}</span>
        </Label>
        <Slider
          min={2}
          max={12}
          step={1}
          value={[settings.rerankTopK]}
          onValueChange={([rerankTopK]) => updateSettings({ rerankTopK })}
        />
      </div>
      <div className="flex items-center justify-between">
        <Label htmlFor="verify">Verify answers</Label>
        <Switch id="verify" checked={settings.verify} onCheckedChange={(verify) => updateSettings({ verify })} />
      </div>
      {!compact && (
        <div className="flex items-center justify-between">
          <Label htmlFor="debug">Pipeline details</Label>
          <Switch id="debug" checked={settings.debug} onCheckedChange={(debug) => updateSettings({ debug })} />
        </div>
      )}
    </div>
  );
}
