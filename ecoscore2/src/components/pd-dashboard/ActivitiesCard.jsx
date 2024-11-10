import React, { useState } from "react";
import { Card, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tv, Laptop, Microwave, ShowerHead, Plus } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import usePostData from "@/hooks/usePostData";
import { url } from "@/data/api";
import { Button } from "../ui/button";

const activities = [
  {
    name: "Watching TV",
    icon: <Tv className="w-6 h-6 text-gray-600" />,
    hours: 2.5,
    appliances: ["Television", "Sound System"],
    energyData: [
      { name: "Television", usage: 150 },
      { name: "Sound System", usage: 50 },
    ],
  },
  {
    name: "Working",
    icon: <Laptop className="w-6 h-6 text-gray-600" />,
    hours: 8,
    appliances: ["Laptop", "Desk Lamp", "Fan"],
    energyData: [
      { name: "Laptop", usage: 65 },
      { name: "Desk Lamp", usage: 40 },
      { name: "Fan", usage: 55 },
    ],
  },
  {
    name: "Cooking",
    icon: <Microwave className="w-6 h-6 text-gray-600" />,
    hours: 1.5,
    appliances: ["Microwave", "Electric Stove", "Hood"],
    energyData: [
      { name: "Microwave", usage: 1100 },
      { name: "Electric Stove", usage: 2000 },
      { name: "Hood", usage: 150 },
    ],
  },
  {
    name: "Bathing",
    icon: <ShowerHead className="w-6 h-6 text-gray-600" />,
    hours: 0.5,
    appliances: ["Geyser"],
    energyData: [{ name: "Geyser", usage: 3000 }],
  },
];

const ActivityDialog = ({ activity, open, onClose }) => {
  if (!activity) return null;

  const totalUsage = activity.energyData.reduce(
    (sum, item) => sum + item.usage,
    0
  );
  const avgUsage = (totalUsage / activity.energyData.length).toFixed(1);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {activity.icon}
            <span>{activity.name} Energy Usage</span>
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-6">
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={activity.energyData}>
                <XAxis dataKey="name" />
                <YAxis
                  label={{
                    value: "Energy Usage (Watts)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                />
                <Tooltip />
                <Bar dataKey="usage" fill="#22c55e" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-4">
            <div className="text-sm space-y-2">
              <p className="font-medium">
                Activity Duration: {activity.hours} hours
              </p>
              <p className="font-medium">
                Total Power Draw: {totalUsage} watts
              </p>
              <p className="font-medium">
                Average Appliance Usage: {avgUsage} watts
              </p>
            </div>

            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-semibold text-sm text-green-800 mb-2">
                Energy Saving Tips:
              </h4>
              <ul className="text-sm text-green-700 list-disc list-inside space-y-1">
                {activity.name === "Watching TV" && (
                  <>
                    <li>Adjust TV brightness to room lighting</li>
                    <li>Use TV's built-in speakers when possible</li>
                    <li>Enable power-saving mode</li>
                  </>
                )}
                {activity.name === "Working" && (
                  <>
                    <li>Use natural light when possible</li>
                    <li>Enable power-saving settings on your laptop</li>
                    <li>Use LED desk lamps</li>
                  </>
                )}
                {activity.name === "Cooking" && (
                  <>
                    <li>Use microwave for smaller portions</li>
                    <li>Keep appliances clean for better efficiency</li>
                    <li>Use lids while cooking to reduce time</li>
                  </>
                )}
                {activity.name === "Bathing" && (
                  <>
                    <li>Reduce shower time</li>
                    <li>Install a low-flow shower head</li>
                    <li>Set optimal water temperature</li>
                  </>
                )}
              </ul>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

const AddActivityDialog = ({ open, onClose, onSubmit }) => {
  const [formData, setFormData] = useState({
    name: "",
    type: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Submitting form data:", formData);
    await onSubmit(formData);
    console.log("Request sent");
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Add New Activity</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Activity Name
            </label>
            <input
              name="name"
              value={formData.name}
              onChange={handleChange}
              className="w-full p-2 border rounded"
              placeholder="Enter activity name"
              required
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Activity Type
            </label>
            <input
              name="type"
              value={formData.type}
              onChange={handleChange}
              className="w-full p-2 border rounded"
              placeholder="Enter activity type"
              required
            />
          </div>
          <Button
            type="submit"
            className="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-[1.02]"
          >
            Add Activity
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default function ActivitiesCard() {
  const [selectedActivity, setSelectedActivity] = useState(null);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const { data, loading, error, postData } = usePostData(url + "/activities");

  const handleAddActivity = async (newActivity) => {
    console.log("Submitting new activity data:", newActivity);
    await postData(newActivity);
    console.log("New activity added");
  };

  const handleActivityClick = (activity) => {
    setSelectedActivity(activity);
  };

  return (
    <div className="w-full h-[35%] bg-inherit p-2">
      <Card className="border">
        <CardTitle className="flex p-4">
          Activities
          <div
            className="mx-2 rounded-full bg-black hover:cursor-pointer hover:shadow-lg"
            onClick={() => setIsAddDialogOpen(true)}
          >
            <Plus className="text-white font-bold" />
          </div>
        </CardTitle>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4">
          {activities.map((activity, index) => (
            <div
              key={index}
              className="bg-white bg-gradient-to-b from-green-100 to-green-200 p-4 rounded-lg shadow-sm hover:shadow-md hover:cursor-pointer transition-shadow"
              onClick={() => handleActivityClick(activity)}
            >
              <div className="flex items-center gap-2 mb-2">
                {activity.icon}
                <h3 className="font-semibold">{activity.name}</h3>
              </div>
              <p className="text-sm text-gray-600">{activity.hours} hours</p>
              <div className="mt-2">
                <p className="text-xs text-gray-500">Appliances:</p>
                <p className="text-xs">{activity.appliances.join(", ")}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <ActivityDialog
        activity={selectedActivity}
        open={!!selectedActivity}
        onClose={() => setSelectedActivity(null)}
      />

      <AddActivityDialog
        open={isAddDialogOpen}
        onClose={() => setIsAddDialogOpen(false)}
        onSubmit={handleAddActivity}
      />
    </div>
  );
}
