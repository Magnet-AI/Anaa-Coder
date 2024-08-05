"use client";

import React, { useState, useRef, useEffect } from "react";
import { IconButton, Box } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import { Input } from "@/components/ui/input";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { darcula } from "react-syntax-highlighter/dist/esm/styles/prism";
import LoadingScreen from "@/components/loading-screen.tsx";

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false); // Add loading state
  const endOfMessagesRef = useRef(null);

  const handleSend = async () => {
    if (input.trim() === "") return;

    const userMessage = { sender: "user", text: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setLoading(true); // Show loading screen

    try {
      const response = await fetch("http://127.0.0.1:8000/llm/response/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();

      // Assuming the API returns a message field in its response
      const botMessage = {
        sender: "bot",
        text: data.final_code || "No response from server.",
      };

      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: "Here's the code snippet for you:" },
        { sender: "bot", text: `\`\`\`python\n${botMessage.text}\n\`\`\`` },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = {
        sender: "bot",
        text: "An error occurred. Please try again.",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false); // Hide loading screen
    }

    setInput("");
  };

  const handleUpload = () => {
    // Add your file upload handling logic here
  };

  const renderMessage = (msg) => {
    if (msg.text.includes("```python")) {
      const code = msg.text.replace(/```python|```/g, "");
      return (
        <SyntaxHighlighter language="python" style={darcula} className="w-full">
          {code}
        </SyntaxHighlighter>
      );
    }
    return msg.text;
  };

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <Box className="relative flex flex-col h-screen border border-gray-300 p-5 bg-gray-200">
      {loading && <LoadingScreen />}
      <Box className="flex-1 overflow-y-auto mb-2.5">
        {messages.map((msg, index) => (
          <Box
            key={index}
            // sx={{ textAlign: msg.sender === "bot" ? "left" : "right" }}
          >
            <Box
              sx={{
                display: "inline-block",
                padding: "10px",
                margin: "10px",
                borderRadius: "5px",
                backgroundColor: msg.sender === "bot" ? "#eee" : "#007bff",
                color: msg.sender === "bot" ? "#000" : "#fff",
                fontSize: "0.85rem",
              }}
            >
              {renderMessage(msg)}
            </Box>
          </Box>
        ))}
        <div ref={endOfMessagesRef} />
      </Box>
      <Box className="flex items-center border border-gray-500 rounded bg-gray-200 p-2 h-12">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask TwoCube"
          type="text"
          className="flex-1 border-none text-gray-700"
        />
        <IconButton onClick={handleUpload}>
          <AttachFileIcon />
        </IconButton>
        <IconButton onClick={handleSend}>
          <SendIcon />
        </IconButton>
      </Box>
    </Box>
  );
};

export default Chat;
