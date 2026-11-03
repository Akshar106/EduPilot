import type { Metadata } from "next";

import { MaterialsPage } from "@/components/materials-page";

export const metadata: Metadata = { title: "Course materials — EduPilot" };

export default function Page() {
  return <MaterialsPage />;
}
