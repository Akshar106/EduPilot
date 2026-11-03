"use client";

import { Loader2, Upload } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import type { Attachment } from "@/components/chat/composer";
import { useApp } from "@/components/providers";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { api } from "@/lib/api";
import { fileSize } from "@/lib/format";
import { summarize, waitForJobs } from "@/lib/jobs";
import type { QueuedUpload } from "@/lib/types";

const ACCEPT = ".pdf,.txt,.md,.docx";

interface AttachDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAttached: (attachments: Attachment[]) => void;
}

/**
 * Add documents to a course's knowledge base and attach them to the next
 * questions, which are then answered from those files. Writing to the shared
 * corpus is admin-only outside development.
 */
export function AttachDialog({ open, onOpenChange, onAttached }: AttachDialogProps) {
  const { config } = useApp();
  const domains = Object.entries(config?.domains ?? {});
  const [domain, setDomain] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const target = domain || domains[0]?.[0] || "";

  const upload = async () => {
    if (!files.length || !target) return;
    setBusy(true);
    const form = new FormData();
    form.append("domain", target);
    for (const f of files) form.append("files", f);
    try {
      const res = await api.upload<{ queued: QueuedUpload[] }>("/api/kb/upload", form);
      // Indexing runs in the background; wait for it, since the files are only
      // searchable once it finishes.
      const jobs = await waitForJobs(res.queued.map((q) => q.job_id));
      const { ok, errors } = summarize(jobs);
      for (const e of errors) toast.error(e);
      const indexed = jobs.filter((j) => j.status === "done");
      if (ok) {
        toast.success(ok);
        onAttached(indexed.map((j) => ({ filename: j.filename, domain: target })));
        setFiles([]);
        onOpenChange(false);
      }
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Attach a document</DialogTitle>
          <DialogDescription>
            It is indexed into the chosen course, and your next questions are answered from it.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="attach-domain">Course</Label>
            <Select value={target} onValueChange={setDomain}>
              <SelectTrigger id="attach-domain" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {domains.map(([abbr, d]) => (
                  <SelectItem key={abbr} value={abbr}>
                    {abbr} — {d.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <label className="flex cursor-pointer flex-col items-center gap-1 rounded-xl border border-dashed p-6 text-sm text-muted-foreground hover:bg-muted/50">
            <Upload className="size-5" />
            Choose files <span className="text-xs">PDF · TXT · MD · DOCX</span>
            <input
              type="file"
              multiple
              accept={ACCEPT}
              className="sr-only"
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
            />
          </label>
          {files.length > 0 && (
            <ul className="space-y-1 text-sm">
              {files.map((f) => (
                <li key={f.name} className="flex justify-between gap-2">
                  <span className="truncate">{f.name}</span>
                  <span className="shrink-0 text-muted-foreground">{fileSize(f.size)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={upload} disabled={busy || !files.length}>
            {busy && <Loader2 className="animate-spin" />} Index and attach
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
