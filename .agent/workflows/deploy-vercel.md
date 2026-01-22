---
description: Deploy frontend to Vercel
---

# Deploy Frontend to Vercel

This workflow guides you through deploying the ICT Impact Dashboard frontend to Vercel.

## Prerequisites

- GitHub account
- Backend is deployed (or you have the backend URL ready)
- Vercel account (free tier works)

## Step 0: Fork the Repository (If You Don't Own It)

**Important:** If you don't own the original repository, you need to fork it first.

1. **Go to the original repository:**
   - Visit: https://github.com/Rahul-Sanskar/ict-impact-dashboard

2. **Fork the repository:**
   - Click the **"Fork"** button in the top-right corner
   - This creates a copy under your GitHub account: `https://github.com/YOUR-USERNAME/ict-impact-dashboard`

3. **Update your local repository to point to your fork:**
   ```bash
   cd c:\Users\agb83\Documents\Thesisly\17 Rahul\ict-impact-dashboard
   
   # Change the origin to your fork (replace YOUR-USERNAME)
   git remote set-url origin https://github.com/YOUR-USERNAME/ict-impact-dashboard.git
   
   # Verify the change
   git remote -v
   # Should show: origin  https://github.com/YOUR-USERNAME/ict-impact-dashboard.git
   ```

4. **Keep the original repo as upstream (optional, for syncing later):**
   ```bash
   git remote add upstream https://github.com/Rahul-Sanskar/ict-impact-dashboard.git
   ```

**Why fork?**
- ✅ Full control over deployments
- ✅ Automatic deployments when you push changes
- ✅ Can make changes without affecting the original repo
- ✅ Can still sync with original repo if needed

## Step 1: Ensure Code is Pushed to GitHub

First, make sure all your latest changes are committed and pushed:

```bash
cd c:\Users\agb83\Documents\Thesisly\17 Rahul\ict-impact-dashboard
git status
git add .
git commit -m "Prepare for Vercel deployment"
git push origin main
```

## Step 2: Sign Up / Log In to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click "Sign Up" or "Log In"
3. Choose "Continue with GitHub"
4. Authorize Vercel to access your GitHub account

## Step 3: Import Your Project

1. Once logged in, click **"Add New..."** → **"Project"**
2. You'll see a list of your GitHub repositories
3. Find **"ict-impact-dashboard"** and click **"Import"**
4. Vercel will detect it's a monorepo with multiple directories

## Step 4: Configure Project Settings

When configuring the project, set these values:

**Framework Preset:**
- Select: **Vite**

**Root Directory:**
- Click **"Edit"** next to Root Directory
- Select: **frontend**
- This tells Vercel to only build the frontend folder

**Build and Output Settings:**
- Build Command: `npm run build` (should be auto-detected)
- Output Directory: `dist` (should be auto-detected)
- Install Command: `npm install` (should be auto-detected)

## Step 5: Configure Environment Variables

Before deploying, add your environment variables:

1. Scroll down to **"Environment Variables"** section
2. Add the following variable:

**For Development/Testing (if backend is local):**
```
Name: VITE_API_URL
Value: http://localhost:8000
```

**For Production (if backend is deployed on Render):**
```
Name: VITE_API_URL
Value: https://your-backend-url.onrender.com
```

**Important Notes:**
- Replace `your-backend-url.onrender.com` with your actual Render backend URL
- Make sure the URL starts with `https://` for production
- Do NOT include a trailing slash

3. Click **"Add"** to save the environment variable

## Step 6: Deploy

1. Click **"Deploy"** button
2. Vercel will:
   - Clone your repository
   - Install dependencies
   - Build your React app
   - Deploy to their CDN
3. Wait 1-2 minutes for the build to complete

## Step 7: Verify Deployment

Once deployment is complete:

1. Vercel will show you a success screen with your deployment URL
2. Click **"Visit"** to open your deployed app
3. Your app will be available at: `https://your-project-name.vercel.app`

**Test the following:**
- ✅ Homepage loads correctly
- ✅ Can navigate between pages
- ✅ Login/signup forms appear
- ✅ Check browser console for errors (F12 → Console)

## Step 8: Update Backend CORS Settings

If your backend is deployed, you need to allow requests from your Vercel domain:

1. Go to your backend hosting platform (Render, etc.)
2. Add your Vercel URL to the `CORS_ORIGINS` environment variable:
   ```
   CORS_ORIGINS=https://your-project-name.vercel.app,https://www.your-project-name.vercel.app
   ```
3. Redeploy your backend for changes to take effect

## Step 9: Test Full Integration

1. Visit your Vercel URL
2. Try to sign up for a new account
3. Try to log in
4. Submit some test data
5. Check if API calls work (check Network tab in DevTools)

**If you see CORS errors:**
- Double-check the `CORS_ORIGINS` setting on your backend
- Make sure `VITE_API_URL` matches your backend URL exactly

## Step 10: Set Up Custom Domain (Optional)

If you have a custom domain:

1. In Vercel dashboard, go to your project
2. Click **Settings** → **Domains**
3. Click **"Add"**
4. Enter your domain name
5. Follow Vercel's instructions to update your DNS records
6. Wait for DNS propagation (can take up to 48 hours)

## Automatic Deployments

Vercel automatically deploys when you push to GitHub:

- **Push to `main` branch** → Deploys to production
- **Push to other branches** → Creates preview deployments
- **Pull requests** → Creates preview deployments with unique URLs

## Updating Your Deployment

To update your deployed app:

```bash
# Make your changes
git add .
git commit -m "Your update message"
git push origin main
```

Vercel will automatically detect the push and redeploy within 1-2 minutes.

## Troubleshooting

### Build Fails

**Check build logs in Vercel:**
1. Go to Vercel dashboard → Your project → Deployments
2. Click on the failed deployment
3. Check the build logs for errors

**Common issues:**
- Missing dependencies: Check `package.json`
- TypeScript errors: Run `npm run build` locally first
- Environment variables: Make sure `VITE_API_URL` is set

### App Loads But API Calls Fail

**Check these:**
1. Open browser DevTools (F12) → Network tab
2. Try to login or make an API call
3. Look for failed requests (red)
4. Check the error message

**Common fixes:**
- Update `VITE_API_URL` in Vercel environment variables
- Update `CORS_ORIGINS` on your backend
- Make sure backend is running and accessible

### Environment Variables Not Working

**Remember:**
- Environment variables starting with `VITE_` are exposed to the browser
- After changing environment variables, you must redeploy
- Click **"Redeploy"** in Vercel dashboard → Deployments

## Useful Vercel Commands (Optional)

You can also deploy using Vercel CLI:

```bash
# Install Vercel CLI globally
npm install -g vercel

# Login to Vercel
vercel login

# Deploy from frontend directory
cd frontend
vercel

# Deploy to production
vercel --prod
```

## Summary

✅ Your frontend is now deployed to Vercel
✅ Automatic deployments on git push
✅ Free SSL certificate
✅ Global CDN for fast loading
✅ Preview deployments for testing

**Your deployment URL:** Check Vercel dashboard for the exact URL

---

**Need help?** Check the [Vercel documentation](https://vercel.com/docs) or the project README.md
