import type { Metadata } from "next";

import { StudyPage } from "@/components/study/study-page";

export const metadata: Metadata = { title: "Self Study — EduPilot" };

export default function Page() {
  return <StudyPage />;
}
