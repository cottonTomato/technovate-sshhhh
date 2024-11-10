import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea"; // Import the Textarea component
import PdSidebar from "@/components/pd-dashboard/PdSidebar";

const JoinCompetitionForm = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const competitionName = location.state?.competitionName || "Competition";

  const [formData, setFormData] = useState({
    fullname: "",
    locality: "",
    agreeToTerms: false,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Save formData to JSON file (this would typically be done on the server-side)
    // For this example, we'll just log it to the console
    console.log(formData);

    // Redirect to leaderboard
    navigate("/leaderboard");
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex">
      <PdSidebar />
      <div className="flex-1 p-8">
        <div className="container mx-auto">
          <Card className="max-w-lg mx-auto">
            <CardHeader>
              <CardTitle>Join {competitionName}</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Full Name
                  </label>
                  <Textarea
                    name="fullname"
                    value={formData.fullname}
                    rows={1}
                    onChange={handleChange}
                    className="mt-2 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-500 focus:ring-opacity-50"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Locality
                  </label>
                  <Textarea
                    name="locality"
                    value={formData.locality}
                    onChange={handleChange}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-500 focus:ring-opacity-50"
                    required
                  />
                </div>
                <div className="flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="agreeToTerms"
                      name="agreeToTerms"
                      type="checkbox"
                      checked={formData.agreeToTerms}
                      onChange={handleChange}
                      className="focus:ring-green-500 h-4 w-4 text-green-600 border-gray-300 rounded"
                      required
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label htmlFor="agreeToTerms" className="font-medium text-gray-700 dark:text-gray-300">
                      I commit to being honest and truthful about the values I give the organizers
                    </label>
                  </div>
                </div>
                <Button type="submit" className="w-full bg-green-600 hover:bg-green-700">
                  Submit
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default JoinCompetitionForm;