import { useState } from 'react';
import axios from 'axios';
import { useAuth } from '@clerk/clerk-react';

const usePostData = (url) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const {getToken} = useAuth()


  const postData = async (postData) => {
    setLoading(true);
    const token = await getToken()
    try {
      const response = await axios.post(url, postData, {
        headers: {Authorization: `Bearer ${token}`,}
      });
      setData(response.data);
    } catch (error) {
      setError(error);
    } finally {
      setLoading(false);
    }
  };

  return { data, loading, error, postData };
};

export default usePostData;