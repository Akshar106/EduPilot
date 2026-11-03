"use client";

import { BookOpen, GraduationCap, Library, MessageSquare, Moon, Sun, UserRound } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";

import { useApp } from "@/components/providers";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/study/", label: "Self Study", icon: BookOpen },
  { href: "/materials/", label: "Course Materials", icon: Library },
  { href: "/profile/", label: "My Progress", icon: UserRound },
];

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
      onClick={() => setTheme(dark ? "light" : "dark")}
    >
      <Sun className="hidden dark:block" />
      <Moon className="dark:hidden" />
    </Button>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { configError } = useApp();
  const active = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex h-14 shrink-0 items-center gap-2 border-b bg-background/95 px-3 backdrop-blur sm:px-4">
        <Link href="/" className="flex items-center gap-2 pr-2 font-semibold">
          <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <GraduationCap className="size-5" />
          </span>
          <span className="hidden sm:inline">EduPilot</span>
        </Link>
        <nav className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
          {NAV.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
                active(href) && "bg-muted font-medium text-foreground",
              )}
            >
              <Icon className="size-4" />
              <span className="hidden md:inline">{label}</span>
            </Link>
          ))}
        </nav>
        <ThemeToggle />
      </header>
      {configError && (
        <div className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">
          Could not reach the EduPilot API: {configError}
        </div>
      )}
      <main className="min-h-0 flex-1">{children}</main>
    </div>
  );
}
