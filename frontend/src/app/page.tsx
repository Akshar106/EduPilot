import { Suspense } from "react";

import { ChatPage } from "@/components/chat/chat-page";

export default function Home() {
  // ChatPage reads the query string, which a static export only knows in the browser.
  return (
    <Suspense>
      <ChatPage />
    </Suspense>
  );
}
