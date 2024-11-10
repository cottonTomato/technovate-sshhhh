import React from 'react';
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";

const PDFDownloader = () => {
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = '/report.pdf';  
    
    const date = new Date().toISOString().split('T')[0];
    link.download = `/public/family_eco_report_20241110.pdf`;
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Button
      onClick={handleDownload}
      className="bg-red-500 hover:bg-red-600 text-white flex items-center gap-2"
    >
      <Download className="h-4 w-4" />
      Download Report
    </Button>
  );
};

export default PDFDownloader;