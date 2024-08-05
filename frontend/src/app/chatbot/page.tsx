import React from "react";
import Sidebar from "../../components/sidebar.tsx";
import Chat from "../../components/chat";
import ProjectSettings from "../../components/project-settings.tsx";
import "../globals.css";

const ChatbotPage = () => {
  return (
    <div className="flex h-screen bg-gray-100">
      <div className="w-1/6 border-r-4 border-gray-300">
        <Sidebar />
      </div>
      <div className="w-1/2 border-r-4 border-gray-100">
        <Chat />
      </div>
      <div className="w-1/3 border-r-4 border-gray-500">
        <ProjectSettings />
      </div>
    </div>
  );
};

export default ChatbotPage;
