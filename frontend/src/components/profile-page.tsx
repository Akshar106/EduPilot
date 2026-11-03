"use client";

import { Sparkles, Trash2 } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { toast } from "sonner";

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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { Profile } from "@/lib/types";

function since(epochSeconds: number): string {
  const minutes = Math.round((Date.now() / 1000 - epochSeconds) / 60);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (minutes < 1440) return `${Math.round(minutes / 60)}h ago`;
  return `${Math.round(minutes / 1440)}d ago`;
}

/**
 * What the tutor has learned about this student, and the means to wipe it.
 * The profile only shapes how explanations are pitched and which topics
 * "quiz me" picks.
 */
export function ProfilePage() {
  const { config } = useApp();
  const colors = Object.fromEntries(Object.entries(config?.domains ?? {}).map(([k, d]) => [k, d.color]));
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<Profile>("/api/profile")
      .then(setProfile)
      .catch((err: Error) => setError(err.message));
  }, []);

  const clear = async () => {
    try {
      await api.del("/api/profile");
      setProfile((p) => (p ? { ...p, topics: [], weak_topics: [], depth: null, turns: 0 } : p));
      toast.success("Your learning profile was cleared.");
    } catch (err) {
      toast.error((err as Error).message);
    }
  };

  const topics = profile?.topics ?? [];
  const weak = profile?.weak_topics ?? [];

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl space-y-5 px-4 py-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">My progress</h1>
            <p className="text-muted-foreground">
              Topics you’ve asked about and the ones you found hard. Used only to pitch explanations and pick quiz
              topics.
            </p>
          </div>
          {!!topics.length && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <Trash2 /> Clear profile
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Clear your learning profile?</AlertDialogTitle>
                  <AlertDialogDescription>
                    The tutor forgets the topics you’ve studied and your preferred depth. Conversations are kept.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction variant="destructive" onClick={() => void clear()}>
                    Clear
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>

        {error && <p className="text-sm text-destructive">Could not load your profile: {error}</p>}

        <div className="grid gap-3 sm:grid-cols-3">
          <Card size="sm">
            <CardHeader>
              <CardDescription>Questions asked</CardDescription>
              <div className="text-2xl font-semibold">{profile?.turns ?? "–"}</div>
            </CardHeader>
          </Card>
          <Card size="sm">
            <CardHeader>
              <CardDescription>Topics covered</CardDescription>
              <div className="text-2xl font-semibold">{profile ? topics.length : "–"}</div>
            </CardHeader>
          </Card>
          <Card size="sm">
            <CardHeader>
              <CardDescription>Preferred depth</CardDescription>
              <div className="text-2xl font-semibold capitalize">{profile?.depth ?? "—"}</div>
            </CardHeader>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Worth practising</CardTitle>
            <CardDescription>Topics you returned to or found confusing.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {weak.length ? (
              <>
                <div className="flex flex-wrap gap-1.5">
                  {weak.map((t) => (
                    <span key={`${t.domain}|${t.name}`} className="rounded-md border px-2 py-1 text-sm">
                      {t.name} <span className="text-xs text-muted-foreground">{t.domain}</span>
                    </span>
                  ))}
                </div>
                <Button asChild size="sm">
                  <Link href={`/?ask=${encodeURIComponent("Quiz me")}`}>
                    <Sparkles /> Quiz me on these
                  </Link>
                </Button>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">Nothing yet — ask a few questions first.</p>
            )}
          </CardContent>
        </Card>

        {!!topics.length && (
          <Card>
            <CardHeader>
              <CardTitle>All topics</CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-muted-foreground">
                  <tr>
                    <th className="pb-2 font-medium">Topic</th>
                    <th className="pb-2 font-medium">Course</th>
                    <th className="pb-2 text-right font-medium">Asked</th>
                    <th className="pb-2 text-right font-medium">Confused</th>
                    <th className="pb-2 text-right font-medium">Last seen</th>
                  </tr>
                </thead>
                <tbody>
                  {topics.map((t) => (
                    <tr key={`${t.domain}|${t.name}`} className="border-t">
                      <td className="py-1.5">{t.name}</td>
                      <td>
                        <span className="rounded px-1.5 text-xs text-white" style={{ background: colors[t.domain] ?? "#888" }}>
                          {t.domain}
                        </span>
                      </td>
                      <td className="text-right">{t.asks}</td>
                      <td className="text-right">{t.confusion}</td>
                      <td className="text-right text-muted-foreground">{since(t.last_seen)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
