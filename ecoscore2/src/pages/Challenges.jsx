import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Trophy,
  Zap,
  Users,
  Timer,
  Award,
  Wind,
  CloudRain,
} from "lucide-react";
import { useNavigate } from "react-router-dom"; // Import useNavigate for routing
import PdSidebar from "@/components/pd-dashboard/PdSidebar";

const ChallengesCompetitions = () => {
  const [activeTab, setActiveTab] = useState("challenges");
  const [activeChallenge, setActiveChallenge] = useState(null);
  const navigate = useNavigate(); // Initialize useNavigate

  const challenges = [
    {
      id: 1,
      title: "Power Down Week",
      description: "Reduce your electricity usage by 20% for one week. ",
      difficulty: "Easy",
      timeFrame: "7 days",
      currentProgress: 65,
      icon: <Zap className="h-6 w-6 text-yellow-500" />,
      targetMetric: "Target: 80 kWh",
      currentMetric: "Current: 52 kWh",
    },
    {
      id: 2,
      title: "Zero Emission Day",
      description:
        "Go one full day without using any carbon-emitting appliances",
      difficulty: "Medium",
      timeFrame: "24 hours",
      currentProgress: 30,
      icon: <Wind className="h-6 w-6 text-blue-500" />,
      targetMetric: "Target: 0 CO2",
      currentMetric: "Current: 2.4 kg CO2",
    },
    {
      id: 3,
      title: "Rain Water Hero",
      description: "Use only collected rainwater for your garden for a month",
      difficulty: "Hard",
      timeFrame: "30 days",
      currentProgress: 45,
      icon: <CloudRain className="h-6 w-6 text-blue-700" />,
      targetMetric: "Target: 100% rainwater",
      currentMetric: "Current: 45%",
    },
  ];

  const competitions = [
    {
      id: 1,
      title: "Neighborhood Energy Challenge",
      description:
        "Compete with similar households in your area for lowest energy consumption",
      participants: 24,
      timeLeft: "5 days",
      prize: "₹500 worth electricity bill coupons",
      currentRank: 3,
      progress: 78,
      icon: <Trophy className="h-6 w-6 text-amber-500" />,
    },
    {
      id: 2,
      title: "Family Eco Warriors",
      description:
        "Challenge other 4-member families to reduce carbon footprint. Start a family challenge!",
      participants: 15,
      timeLeft: "12 days",
      prize: "₹1000 solar product vouchers",
      currentRank: 5,
      progress: 62,
      icon: <Users className="h-6 w-6 text-green-600" />,
    },
  ];

  return (
    <div className="min-h-screen h-screen overflow-hidden bg-gray-50 dark:bg-gray-900 flex">
      <PdSidebar />
      <div className="flex-1 p-8">
        <div className="container mx-auto">
          <div className="flex flex-col items-center space-y-4">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Challenges & Competitions
            </h1>
            <div className="grid w-full max-w-md grid-cols-2 gap-4">
              <Button
                variant={activeTab === "challenges" ? "solid" : "outline"}
                onClick={() => setActiveTab("challenges")}
                className={
                  activeTab === "challenges"
                    ? "bg-green-600 text-white"
                    : "bg-white text-green-600 border-green-600"
                }
              >
                Self Challenges
              </Button>
              <Button
                variant={activeTab === "competitions" ? "solid" : "outline"}
                onClick={() => setActiveTab("competitions")}
                className={
                  activeTab === "competitions"
                    ? "bg-green-600 text-white"
                    : "bg-white text-green-600 border-green-600"
                }
              >
                Competitions
              </Button>
            </div>
          </div>

          {/* Content */}
          <div className="space-y-8 mt-8">
            {activeTab === "challenges" && (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {challenges.map((challenge) => (
                  <Card
                    key={challenge.id}
                    className="hover:shadow-lg transition-shadow"
                  >
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        {challenge.icon}
                        <Badge
                          variant={
                            challenge.difficulty === "Easy"
                              ? "success"
                              : challenge.difficulty === "Medium"
                              ? "warning"
                              : "destructive"
                          }
                        >
                          {challenge.difficulty}
                        </Badge>
                      </div>
                      <CardTitle className="mt-4">{challenge.title}</CardTitle>
                      <CardDescription>{challenge.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">
                          <Timer className="h-4 w-4 inline mr-1" />
                          {challenge.timeFrame}
                        </span>
                        <span className="text-gray-600 dark:text-gray-400">
                          Progress
                        </span>
                      </div>
                      <Progress
                        value={challenge.currentProgress}
                        className="h-2"
                      />
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">
                          {challenge.targetMetric}
                        </span>
                        <span className="text-gray-600 dark:text-gray-400">
                          {challenge.currentMetric}
                        </span>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button
                        className="w-full bg-green-600 hover:bg-green-700"
                        onClick={() => setActiveChallenge(challenge.id)}
                      >
                        {activeChallenge === challenge.id
                          ? "Participating"
                          : "Join Challenge"}
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            )}

            {activeTab === "competitions" && (
              <div className="grid md:grid-cols-2 gap-6">
                {competitions.map((competition) => (
                  <Card
                    key={competition.id}
                    className="hover:shadow-lg transition-shadow"
                  >
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        {competition.icon}
                        <Badge
                          variant="outline"
                          className="bg-amber-100 dark:bg-amber-900"
                        >
                          <Award className="h-4 w-4 mr-1" />
                          Prize Pool
                        </Badge>
                      </div>
                      <CardTitle className="mt-4">
                        {competition.title}
                      </CardTitle>
                      <CardDescription>
                        {competition.description}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded-lg">
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            Participants
                          </div>
                          <div className="text-lg font-semibold flex items-center gap-1">
                            <Users className="h-4 w-4" />
                            {competition.participants}
                          </div>
                        </div>
                        <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded-lg">
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            Time Left
                          </div>
                          <div className="text-lg font-semibold flex items-center gap-1">
                            <Timer className="h-4 w-4" />
                            {competition.timeLeft}
                          </div>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">
                            Your Rank: #{competition.currentRank}
                          </span>
                          <span className="text-gray-600 dark:text-gray-400">
                            Progress
                          </span>
                        </div>
                        <Progress
                          value={competition.progress}
                          className="h-2"
                        />
                      </div>
                      <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg">
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          Prize
                        </div>
                        <div className="text-green-600 dark:text-green-400 font-medium">
                          {competition.prize}
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button
                        className="w-full bg-green-600 hover:bg-green-700"
                        onClick={() => navigate("/join-competition")}
                      >
                        Join Competition
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChallengesCompetitions;
