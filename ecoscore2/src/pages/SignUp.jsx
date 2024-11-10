import { SignUp as ClerkSignUp } from "@clerk/clerk-react";

export default function SignUp() {
  return (
    <div className="flex justify-center items-center h-screen w-full">
      <ClerkSignUp redirectUrl="/onboarding" signInUrl="/sign-in" />
    </div>
  );
}
