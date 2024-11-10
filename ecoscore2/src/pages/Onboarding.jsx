import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Leaf } from "lucide-react";
import usePostData from "@/hooks/usePostData";
import { url } from "@/data/api";

const Onboarding = () => {
  const navigate = useNavigate();
  const { data, loading, error, postData } = usePostData(url + "/user");

  const [formData, setFormData] = useState({
    firstname: "",
    lastname: "",
    agreeToTerms: false,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Submitting form data:", formData);
    await postData({
      firstname: formData.firstname,
      lastname: formData.lastname,
    });
    console.log("Request sent, navigating to dashboard");
    navigate("/dashboard");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl mx-auto shadow-xl">
        <CardHeader className="text-center space-y-4 pb-8">
          <div className="flex items-center justify-center">
            <Leaf className="h-12 w-12 text-green-600" />
          </div>
          <CardTitle className="text-3xl font-bold text-gray-900 dark:text-white">
            Welcome to EcoScore
          </CardTitle>
          <p className="text-gray-600 dark:text-gray-300">
            Join us in making a difference
          </p>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  First Name
                </label>
                <Textarea
                  name="firstname"
                  value={formData.firstname}
                  rows={1}
                  onChange={handleChange}
                  className="w-full transition-all duration-200 focus:ring-2 focus:ring-green-500/50 resize-none"
                  placeholder="Enter your first name"
                  required
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Last Name
                </label>
                <Textarea
                  name="lastname"
                  value={formData.lastname}
                  rows={1}
                  onChange={handleChange}
                  className="w-full transition-all duration-200 focus:ring-2 focus:ring-green-500/50 resize-none"
                  placeholder="Enter your last name"
                  required
                />
              </div>
            </div>

            <div className="flex items-start space-x-3 pt-4">
              <input
                id="agreeToTerms"
                name="agreeToTerms"
                type="checkbox"
                checked={formData.agreeToTerms}
                onChange={handleChange}
                className="mt-1 h-4 w-4 rounded border-gray-300 text-green-600 focus:ring-green-500 transition-all duration-200"
                required
              />
              <label
                htmlFor="agreeToTerms"
                className="text-sm text-gray-600 dark:text-gray-300"
              >
                I agree to the{" "}
                <a
                  href="#"
                  className="text-green-600 hover:text-green-700 underline"
                >
                  terms and conditions
                </a>
              </label>
            </div>

            <Button
              type="submit"
              className="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-[1.02]"
              disabled={loading}
            >
              {loading ? "Submitting..." : "Get Started"}
            </Button>
            {error && (
              <p className="text-red-500 text-center mt-4">{error.message}</p>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default Onboarding;
