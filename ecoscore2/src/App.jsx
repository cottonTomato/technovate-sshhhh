import { Route, Routes, BrowserRouter as Router } from "react-router-dom";
import { SignedIn, SignedOut, RedirectToSignIn } from "@clerk/clerk-react";
import Dashboard from "@/pages/Dashboard";
import Onboarding from "@/pages/Onboarding";
import LandingPage from "@/pages/LandingPage";
import SignIn from "@/pages/SignIn";
import SignUp from "@/pages/SignUp";
import AnalyticsDash from "@/pages/Analytics";
import ChallengesCompetitions from "./pages/Challenges";
import Leaderboard from "./pages/Leaderboard";
import JoinCompetitionForm from "./pages/JoinCompetitionForm";
import ChatbotApp from "./pages/Chatbot-page";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route
          path="/dashboard"
          element={
            <>
              <SignedIn>
                <Dashboard />
              </SignedIn>
              <SignedOut>
                <RedirectToSignIn redirectUrl="/sign-in" />
              </SignedOut>
            </>
          }
        />
        <Route
          path="/onboarding"
          element={
            <>
              <SignedIn>
                <Onboarding />
              </SignedIn>
              <SignedOut>
                <RedirectToSignIn redirectUrl="/sign-in" />
              </SignedOut>
            </>
          }
        />
        <Route path="/sign-in" element={<SignIn />} />
        <Route path="/sign-up" element={<SignUp />} />
        <Route
          path="/analytics"
          element={
            <>
              <SignedIn>
                <AnalyticsDash />
              </SignedIn>
              <SignedOut>
                <RedirectToSignIn redirectUrl="/sign-in" />
              </SignedOut>
            </>
          }
        />
        <Route
          path="/challenges"
          element={
            <>
              <SignedIn>
                <ChallengesCompetitions />
              </SignedIn>
              <SignedOut>
                <RedirectToSignIn redirectUrl="/sign-in" />
              </SignedOut>
            </>
          }
        />
        <Route
          path="/leaderboard"
          element={
            <>
              <SignedIn>
                <Leaderboard />
              </SignedIn>
              <SignedOut>
                <RedirectToSignIn redirectUrl="/sign-in" />
              </SignedOut>
            </>
          }
        />
        <Route
          path="/join-competition"
          element={
            <>
              <SignedIn>
                <JoinCompetitionForm />
              </SignedIn>
              <SignedOut>
                <RedirectToSignIn redirectUrl="/sign-in" />
              </SignedOut>
            </>
          }
        />
        <Route
          path="/chatbot"
          element={
            <>
              <SignedIn>
                <ChatbotApp />
              </SignedIn>
              <SignedOut>
                <RedirectToSignIn redirectUrl="/sign-in" />
              </SignedOut>
            </>
          }
        />
      </Routes>
    </Router>
  );
};

export default App;
