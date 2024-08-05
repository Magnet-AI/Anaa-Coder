import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const ProjectSettings = () => {
  return (
    <div className="flex flex-col w-full h-full bg-gray-100 p-4">
      <h1 className="ml-3 text-2xl font-bold text-black">Twocube's Den</h1>
      <Tabs
        defaultValue="planner"
        className="w-full max-w-md mt-4 text-center text-black"
      >
        <TabsList className="bg-gray-200 p-1 rounded text-center">
          <TabsTrigger
            value="planner"
            className="w-1/2 p-2 text-black hover:bg-gray-300 rounded"
          >
            Planner
          </TabsTrigger>
          <TabsTrigger
            value="terminal"
            className="w-1/2 p-2 text-black hover:bg-gray-300 rounded"
          >
            Terminal
          </TabsTrigger>
          <TabsTrigger
            value="code"
            className="w-1/2 p-2 text-black hover:bg-gray-300 rounded"
          >
            Code
          </TabsTrigger>
        </TabsList>
        <TabsContent
          value="planner"
          className="mt-4 p-4 bg-gray-300 rounded text-black h-full"
        >
          Planner
        </TabsContent>
        <TabsContent
          value="terminal"
          className="mt-4 p-4 bg-gray-300 rounded text-black h-full"
        >
          Terminal
        </TabsContent>
        <TabsContent
          value="code"
          className="mt-4 p-4 bg-gray-300 rounded text-black h-full"
        >
          Code
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ProjectSettings;
